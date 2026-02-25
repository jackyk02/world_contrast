"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from enum import Enum
from enum import auto
import concurrent.futures
import stat
import json
import logging
import pathlib
from pathlib import Path
import shutil
import time
import tqdm
import urllib.parse
import os

import filelock
import gcsfs

# Reduce TensorFlow log spam (must be set before importing tensorflow).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds

logger = logging.getLogger(__name__)


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # NOTE: This buffer stores *decoded frames* (including images) in memory; very large values can OOM,
        # especially under DDP (multiple processes) and when building both train+eval pipelines.
        # Reducing from 10000 to 1000 to save memory in multi-GPU settings.
        shuffle_buffer_size: int = 10000,
        num_parallel_reads: int = 4,  # Reduced from -1 (AUTOTUNE) to save memory
        num_parallel_calls: int = 4,  # Reduced from -1 (AUTOTUNE) to save memory
        filter_dict_path=None,  # Path to json file with indices to sample during training
        rank: int = 0,
        world_size: int = 1,
        target_height: int = 224,
        target_width: int = 224,
    ):

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        if world_size > 1:
            dataset = dataset.shard(world_size, rank)

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        # # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        # Load the filter dictionary if provided.
        # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
        # (e.g.,
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # means keep frames 0-99 and 200-299).
        if filter_dict_path is not None:
            cached_filter_dict_path = self.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
            # Explicitly clear large lists to save memory
            del keys_tensor
            del values_tensor
            del filter_dict
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["joint_position"]
                        if action_space == DroidActionSpace.JOINT_POSITION
                        else traj["action_dict"]["joint_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Use exterior_image_1_left if available, otherwise fall back to exterior_image_2_left
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            exterior_img_1 = traj["observation"]["exterior_image_1_left"]
            exterior_img = tf.cond(
                tf.greater(tf.size(exterior_img_1), 0),
                lambda: exterior_img_1,
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.cond(
                tf.greater(tf.size(traj["language_instruction"]), 0),
                lambda: traj["language_instruction"],
                lambda: tf.cond(
                    tf.greater(tf.size(traj["language_instruction_2"]), 0),
                    lambda: traj["language_instruction_2"],
                    lambda: traj["language_instruction_3"],
                ),
            )


            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))

            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks with history."""
            traj_len = tf.shape(traj["actions"])[0]

            # History length is action_chunk_size - 1
            # For action_chunk_size=4, we want: [a(t-3), a(t-2), a(t-1), a(t), a(t+1), a(t+2), a(t+3)]
            # Total chunk size = history + current + future = (action_chunk_size-1) + 1 + (action_chunk_size-1) = 2*action_chunk_size - 1
            history_length = action_chunk_size - 1
            total_chunk_size = 2 * action_chunk_size - 1
            
            # Create offset indices: [-(action_chunk_size-1), ..., -1, 0, 1, ..., (action_chunk_size-1)]
            offset_indices = tf.range(-history_length, action_chunk_size)
            
            # For each step in the trajectory, construct indices for history + current + future actions
            action_chunk_indices = tf.broadcast_to(
                offset_indices[None],
                [traj_len, total_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, total_chunk_size],
            )

            # Clamp to valid range [0, traj_len-1]
            # Early timesteps will repeat the first action for history
            # Late timesteps will repeat the last action for future
            action_chunk_indices = tf.clip_by_value(action_chunk_indices, 0, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Filter data that doesn't pass the filter
        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_and_resize_images(traj):
            # Decode
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            
            # Resize with aspect ratio preservation (simple version: resize and then pad)
            def resize_and_pad(img):
                img = tf.image.resize(img, [target_height, target_width], method="bilinear", preserve_aspect_ratio=True)
                img = tf.image.resize_with_pad(img, target_height, target_width)
                return tf.cast(tf.round(img), tf.uint8)

            traj["observation"]["image"] = resize_and_pad(traj["observation"]["image"])
            traj["observation"]["wrist_image"] = resize_and_pad(traj["observation"]["wrist_image"])
            return traj

        dataset = dataset.frame_map(decode_and_resize_images, num_parallel_calls)

        # Shuffle, batch
        # IMPORTANT: only shuffle for training. For eval/stable streams (shuffle=False),
        # shuffling wastes memory/time and can trigger large "Filling up shuffle buffer..." logs.
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000
    
    def get_cache_dir(self) -> pathlib.Path:
        cache_dir = pathlib.Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._set_folder_permission(cache_dir)
        return cache_dir
    
    def _set_permission(self, path: pathlib.Path, target_permission: int):
        """chmod requires executable permission to be set, so we skip if the permission is already match with the target."""
        if path.stat().st_mode & target_permission == target_permission:
            print(f"Skipping {path} because it already has correct permissions")
            return
        path.chmod(target_permission)
        print(f"Set {path} to {target_permission}")
        
    def _should_invalidate_cache(self, cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
        """Invalidate the cache if it is expired. Return True if the cache was invalidated."""

        assert local_path.exists(), f"File not found at {local_path}"

        relative_path = str(local_path.relative_to(cache_dir))
        # Cache invalidation rules are optional; if not provided, never invalidate.
        invalidate_rules = getattr(self, "_INVALIDATE_CACHE_DIRS", None)
        if not invalidate_rules:
            return False

        for pattern, expire_time in invalidate_rules.items():
            if pattern.match(relative_path):
                # Remove if not newer than the expiration timestamp.
                return local_path.stat().st_mtime <= expire_time

        return False

    def _set_folder_permission(self, folder_path: pathlib.Path) -> None:
        """Set folder permission to be read, write and searchable."""
        self._set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    def _ensure_permissions(self, path: pathlib.Path) -> None:
        """Ensure cache directory and files have correct permissions for sharing."""

        def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
            cache_dir = self.get_cache_dir()
            relative_path = path.relative_to(cache_dir)
            moving_path = cache_dir
            for part in relative_path.parts:
                self._set_folder_permission(moving_path / part)
                moving_path = moving_path / part

        def _set_file_permission(file_path: pathlib.Path) -> None:
            """Set all files to be read & writable, if it is a script, keep it as a script."""
            file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
            if file_path.stat().st_mode & 0o100:
                self._set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            else:
                self._set_permission(file_path, file_rw)

        _setup_folder_permission_between_cache_dir_and_path(path)
        for root, dirs, files in os.walk(str(path)):
            root_path = pathlib.Path(root)
            for file in files:
                file_path = root_path / file
                _set_file_permission(file_path)

            for dir in dirs:
                dir_path = root_path / dir
                self._set_folder_permission(dir_path)

    def _download_gcsfs(self, url: str, local_path: pathlib.Path, **kwargs) -> None:
        """Download a file from Google Cloud Storage to the local cache using gcsfs."""
        # Parse GCS URL (gs://bucket/path)
        parsed = urllib.parse.urlparse(url)
        bucket = parsed.netloc
        path = parsed.path.lstrip("/")
        gcs_path = f"{bucket}/{path}"

        # Initialize gcsfs with anonymous access for public buckets
        fs = gcsfs.GCSFileSystem(token="anon")
        
        # Check if it's a directory or file
        try:
            info = fs.info(gcs_path)
            is_dir = info["type"] == "directory"
        except FileNotFoundError:
            # Could be a directory prefix
            is_dir = fs.isdir(gcs_path)

        if is_dir:
            total_size = sum(f["size"] for f in fs.ls(gcs_path, detail=True) if f["type"] == "file")
        else:
            total_size = info.get("size", 0)

        with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024, desc=f"Downloading {url}") as pbar:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
            def download_task():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if is_dir:
                    fs.get(gcs_path, str(local_path), recursive=True)
                else:
                    fs.get(gcs_path, str(local_path))
            
            future = executor.submit(download_task)
            while not future.done():
                if local_path.exists():
                    if local_path.is_file():
                        current_size = local_path.stat().st_size
                    else:
                        current_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
                    pbar.update(current_size - pbar.n)
                time.sleep(1)
            # Ensure we got the result (raises any exception that occurred)
            future.result()
            pbar.update(total_size - pbar.n)

    def maybe_download(self, url: str, *, force_download: bool = False, **kwargs) -> pathlib.Path:
        
        """Download a file or directory from a remote filesystem to the local cache, and return the local path.

        If the local file already exists, it will be returned directly.

        It is safe to call this function concurrently from multiple processes.
        See `get_cache_dir` for more details on the cache directory.

        Args:
            url: URL to the file to download.
            force_download: If True, the file will be downloaded even if it already exists in the cache.
            **kwargs: Additional arguments to pass to fsspec.

        Returns:
            Local path to the downloaded file or directory. That path is guaranteed to exist and is absolute.
        """
        # Don't use fsspec to parse the url to avoid unnecessary connection to the remote filesystem.
        parsed = urllib.parse.urlparse(url)

        # Short circuit if this is a local path.
        if parsed.scheme == "":
            path = pathlib.Path(url)
            if not path.exists():
                raise FileNotFoundError(f"File not found at {url}")
            return path.resolve()

        cache_dir = self.get_cache_dir()

        local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
        local_path = local_path.resolve()

        # Check if the cache should be invalidated.
        invalidate_cache = False
        if local_path.exists():
            if force_download or self._should_invalidate_cache(cache_dir, local_path):
                invalidate_cache = True
            else:
                return local_path

        try:
            lock_path = local_path.with_suffix(".lock")
            with filelock.FileLock(lock_path):
                # Ensure consistent permissions for the lock file.
                self._ensure_permissions(lock_path)
                # First, remove the existing cache if it is expired.
                if invalidate_cache:
                    logger.info(f"Removing expired cached entry: {local_path}")
                    if local_path.is_dir():
                        shutil.rmtree(local_path)
                    else:
                        local_path.unlink()

                # Download the data to a local cache.
                logger.info(f"Downloading {url} to {local_path}")
                scratch_path = local_path.with_suffix(".partial")
                self._download_gcsfs(url, scratch_path, **kwargs)

                shutil.move(scratch_path, local_path)
                self._ensure_permissions(local_path)

        except PermissionError as e:
            msg = (
                f"Local file permission error was encountered while downloading {url}. "
                f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
            )
            raise PermissionError(msg) from e

        return local_path
