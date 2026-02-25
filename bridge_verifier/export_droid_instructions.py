#!/usr/bin/env python3
"""
Extract episode-level instructions from droid_1.0.1 into a simple mapping JSON.

Rules:
- Use per-step language_instruction / language_instruction_2 / language_instruction_3 (first non-empty).
- Fall back to meta/tasks.parquet via task_index if language fields are empty.
- Skip episodes with no instruction text after fallback.
- Skip episodes whose trajectory is marked unsuccessful (is_episode_successful == False).
- Output JSON similar to instruction_mapping.json but without rephrases:
    {
      "0": {"original": "put the block on the plate", "episode_id": 0},
      ...
    }
"""

import argparse
import glob
import json
import os
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm


def normalize_instruction(instr: Optional[str]) -> Optional[str]:
    if not instr or not isinstance(instr, str):
        return None
    instr = instr.strip()
    return instr if instr else None


def first_non_empty(series: pd.Series) -> Optional[str]:
    for val in series:
        if isinstance(val, str) and val.strip():
            return normalize_instruction(val)
    return None


def load_tasks_map(tasks_path: str) -> Dict[int, str]:
    """Return mapping task_index -> normalized text (if available)."""
    if not os.path.exists(tasks_path):
        return {}
    df = pd.read_parquet(tasks_path).reset_index()
    mapping: Dict[int, str] = {}
    for _, row in df.iterrows():
        text = row.get("index", None)
        norm = normalize_instruction(text) if isinstance(text, str) else None
        if norm is not None:
            mapping[int(row["task_index"])] = norm
    return mapping


def main(dataset_root: str, output_json: str):
    tasks_map = load_tasks_map(os.path.join(dataset_root, "meta", "tasks.parquet"))

    episode_meta_paths = sorted(
        glob.glob(os.path.join(dataset_root, "meta", "episodes", "chunk-*", "file-*.parquet"))
    )
    episodes_jsonl = os.path.join(dataset_root, "meta", "episodes.jsonl")
    use_jsonl = False
    if not episode_meta_paths:
        if os.path.exists(episodes_jsonl):
            use_jsonl = True
        else:
            raise FileNotFoundError("No episode metadata found under meta/episodes or meta/episodes.jsonl")

    out: Dict[str, Dict[str, object]] = {}
    next_id = 0

    data_cache: Dict[tuple, pd.DataFrame] = {}

    def load_data(chunk_idx: int, file_idx: int) -> Optional[pd.DataFrame]:
        key = (chunk_idx, file_idx)
        if key in data_cache:
            return data_cache[key]
        path = os.path.join(dataset_root, "data", f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet")
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
        data_cache[key] = df
        return df

    if use_jsonl:
        with open(episodes_jsonl, "r") as f:
            for line in tqdm(f, desc="Episodes (jsonl)"):
                rec = json.loads(line)
                episode_id = rec.get("episode_index")
                tasks = rec.get("tasks", [])
                instr_text = None
                if isinstance(tasks, list):
                    for t in tasks:
                        if isinstance(t, str) and t.strip():
                            instr_text = normalize_instruction(t)
                            if instr_text:
                                break
                if not instr_text:
                    continue
                out[str(next_id)] = {"original": instr_text, "episode_id": episode_id}
                next_id += 1
    else:
        for ep_meta_path in tqdm(episode_meta_paths, desc="Episode files"):
            ep_df = pd.read_parquet(ep_meta_path)
            for _, ep_row in tqdm(ep_df.iterrows(), total=len(ep_df), leave=False, desc="Episodes"):
                episode_id = int(ep_row["episode_index"])
                data_chunk = int(ep_row["data/chunk_index"])
                data_file = int(ep_row["data/file_index"])
                data_from = int(ep_row["dataset_from_index"])
                data_to = int(ep_row["dataset_to_index"])

                df = load_data(data_chunk, data_file)
                if df is None:
                    # Missing or unreadable data file; skip this episode
                    continue
                ep_slice = df.iloc[data_from:data_to]
                if ep_slice.empty:
                    continue

                # Skip unsuccessful episodes
                success = bool(ep_slice["is_episode_successful"].iloc[0])
                if not success:
                    continue

                instr_text = (
                    first_non_empty(ep_slice["language_instruction"])
                    or first_non_empty(ep_slice["language_instruction_2"])
                    or first_non_empty(ep_slice["language_instruction_3"])
                )
                if instr_text is None and "task_index" in ep_slice.columns and len(ep_slice["task_index"]) > 0:
                    instr_text = tasks_map.get(int(ep_slice["task_index"].iloc[0]))
                if instr_text is None:
                    continue

                out[str(next_id)] = {"original": instr_text, "episode_id": episode_id}
                next_id += 1

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} instructions to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export episode instructions from droid_1.0.1 (no rephrases)")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of droid_1.0.1 dataset")
    parser.add_argument("--output_json", type=str, required=True, help="Path to write instruction mapping JSON")
    args = parser.parse_args()
    main(args.dataset_root, args.output_json)

