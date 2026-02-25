#!/usr/bin/env python3
"""
Add DROID dataset's alternative instructions (language_instruction_2, language_instruction_3)
to the top of the rephrase list in the existing rephrase JSON file.

This script:
1. Scans the DROID dataset to extract all 3 instruction variants per trajectory
2. Builds a mapping: language_instruction -> [language_instruction_2, language_instruction_3]
3. Updates the existing rephrase JSON by prepending the alternative instructions

Usage:
 python add_droid_instructions_to_rephrases.py \
    --rlds_data_dir /root/data \
    --input_json filtered_droid_rephrases_16.json \
    --output_json filtered_droid_rephrases_16_with_alternates.json \
    --num_trajectories 100000 \
    --force_rescan
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Set TensorFlow log level before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def normalize_instruction(text: str) -> str:
    """Normalize instruction for matching (lowercase, strip, remove punctuation)."""
    text = text.lower().strip()
    # Remove common punctuation but keep spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_instruction_from_tensor(tensor) -> str:
    """Extract a single instruction string from a tensor (which may be an array)."""
    import numpy as np
    
    arr = np.asarray(tensor)
    if arr.ndim == 0:
        # Scalar
        val = arr.item()
    else:
        # Array - take first element (all should be same within trajectory)
        val = arr.flat[0]
    
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    return val.strip() if val else ""


def build_droid_instruction_mapping(
    data_dir: str,
    num_trajectories: int = 100000,
    save_cache: bool = True,
    cache_path: str = "droid_instruction_mapping_cache.json",
) -> dict[str, list[str]]:
    """
    Scan DROID dataset and build mapping from language_instruction to alternatives.
    
    Returns:
        Dict mapping normalized language_instruction -> list of [lang2, lang3] (non-empty, deduplicated)
    """
    # Check if cache exists
    if save_cache and os.path.exists(cache_path):
        print(f"Loading cached instruction mapping from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)
    
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import dlimp as dl
    
    print(f"Scanning DROID dataset from {data_dir}...")
    print(f"Will process up to {num_trajectories} trajectories")
    
    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=4)
    
    # Only process successful trajectories
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
        )
    )
    
    # Take limited number of trajectories
    dataset = dataset.take(num_trajectories)
    
    # Mapping: normalized_instruction -> set of alternative instructions
    instruction_alternates: dict[str, set[str]] = defaultdict(set)
    
    count = 0
    for traj in tqdm(dataset, desc="Extracting instructions", total=num_trajectories):
        lang1 = extract_instruction_from_tensor(traj["language_instruction"])
        lang2 = extract_instruction_from_tensor(traj["language_instruction_2"])
        lang3 = extract_instruction_from_tensor(traj["language_instruction_3"])
        
        if not lang1:
            continue
        
        # Use normalized version as key for matching
        norm_key = normalize_instruction(lang1)
        
        # Add non-empty alternatives (that are different from the original)
        norm1 = normalize_instruction(lang1)
        if lang2:
            norm2 = normalize_instruction(lang2)
            if norm2 and norm2 != norm1:
                instruction_alternates[norm_key].add(lang2)
        
        if lang3:
            norm3 = normalize_instruction(lang3)
            if norm3 and norm3 != norm1:
                instruction_alternates[norm_key].add(lang3)
        
        count += 1
    
    print(f"Processed {count} trajectories")
    print(f"Found {len(instruction_alternates)} unique instructions with alternatives")
    
    # Convert sets to sorted lists
    result = {k: sorted(list(v)) for k, v in instruction_alternates.items() if v}
    
    # Save cache
    if save_cache:
        print(f"Saving instruction mapping cache to {cache_path}")
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
    
    return result


def update_rephrase_json(
    input_json_path: str,
    output_json_path: str,
    instruction_mapping: dict[str, list[str]],
) -> tuple[int, int]:
    """
    Update the rephrase JSON by prepending alternative instructions.
    
    Args:
        input_json_path: Path to input rephrase JSON
        output_json_path: Path to output JSON
        instruction_mapping: Mapping from normalized instruction -> alternatives
    
    Returns:
        Tuple of (num_entries_updated, total_alternates_added)
    """
    print(f"Loading rephrase JSON from {input_json_path}")
    with open(input_json_path, "r") as f:
        rephrase_data = json.load(f)
    
    entries_updated = 0
    total_alternates_added = 0
    
    for entry_id, entry in tqdm(rephrase_data.items(), desc="Updating rephrases"):
        original = entry.get("original", "")
        if not original:
            continue
        
        # Normalize the original instruction for matching
        norm_original = normalize_instruction(original)
        
        # Check if we have alternatives for this instruction
        if norm_original not in instruction_mapping:
            continue
        
        alternatives = instruction_mapping[norm_original]
        if not alternatives:
            continue
        
        # Get existing rephrases
        existing_rephrases = entry.get("rephrases", [])
        
        # Create set of normalized existing rephrases to avoid duplicates
        existing_normalized = {normalize_instruction(r) for r in existing_rephrases}
        existing_normalized.add(norm_original)  # Don't add if same as original
        
        # Filter alternatives to only include ones not already in the list
        new_alternates = []
        for alt in alternatives:
            norm_alt = normalize_instruction(alt)
            if norm_alt not in existing_normalized:
                new_alternates.append(alt)
                existing_normalized.add(norm_alt)
        
        if new_alternates:
            # Prepend new alternatives to the TOP of the rephrase list
            entry["rephrases"] = new_alternates + existing_rephrases
            entries_updated += 1
            total_alternates_added += len(new_alternates)
    
    print(f"Updated {entries_updated} entries with {total_alternates_added} new alternatives")
    
    # Save updated JSON
    print(f"Saving updated rephrase JSON to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(rephrase_data, f, indent=2)
    
    return entries_updated, total_alternates_added


def main():
    parser = argparse.ArgumentParser(
        description="Add DROID alternative instructions to rephrase JSON"
    )
    parser.add_argument(
        "--rlds_data_dir",
        type=str,
        default="/root/data",
        help="Parent directory of droid/ TFDS folder",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="filtered_droid_rephrases_16.json",
        help="Input rephrase JSON file",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output rephrase JSON file (default: input_with_alternates.json)",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=10000,
        help="Maximum number of trajectories to scan",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="droid_instruction_mapping_cache.json",
        help="Path to cache the instruction mapping",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Don't use/save cache",
    )
    parser.add_argument(
        "--force_rescan",
        action="store_true",
        help="Force rescan even if cache exists",
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_json is None:
        input_path = Path(args.input_json)
        args.output_json = str(input_path.with_stem(input_path.stem + "_with_alternates"))
    
    # Handle cache
    cache_path = args.cache_path if not args.no_cache else None
    if args.force_rescan and cache_path and os.path.exists(cache_path):
        print(f"Removing existing cache: {cache_path}")
        os.remove(cache_path)
    
    # Step 1: Build instruction mapping from DROID
    print("=" * 60)
    print("Step 1: Building instruction mapping from DROID dataset")
    print("=" * 60)
    
    instruction_mapping = build_droid_instruction_mapping(
        data_dir=args.rlds_data_dir,
        num_trajectories=args.num_trajectories,
        save_cache=not args.no_cache,
        cache_path=args.cache_path,
    )
    
    # Print some stats
    total_alternates = sum(len(v) for v in instruction_mapping.values())
    print(f"\nInstruction mapping stats:")
    print(f"  Unique instructions with alternates: {len(instruction_mapping)}")
    print(f"  Total alternative instructions: {total_alternates}")
    
    # Step 2: Update the rephrase JSON
    print("\n" + "=" * 60)
    print("Step 2: Updating rephrase JSON with alternatives")
    print("=" * 60)
    
    entries_updated, alternates_added = update_rephrase_json(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        instruction_mapping=instruction_mapping,
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Input:  {args.input_json}")
    print(f"Output: {args.output_json}")
    print(f"Entries updated: {entries_updated}")
    print(f"Alternatives added: {alternates_added}")


if __name__ == "__main__":
    main()

