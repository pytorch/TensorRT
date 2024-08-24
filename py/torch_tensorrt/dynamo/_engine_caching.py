import copy
import logging
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch._inductor.codecache import FxGraphCachePickler
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily

_LOGGER: logging.Logger = logging.getLogger(__name__)


class BaseEngineCache(ABC):

    @abstractmethod
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    @staticmethod
    def get_hash(gm: torch.fx.GraphModule) -> str:
        """Get the hash value of the GraphModule

        Args:
            gm (torch.fx.GraphModule): GraphModule to hash

        Returns:
            str: hash value of the GraphModule
        """
        # parameters are set to 0
        with unset_fake_temporarily():
            new_gm = copy.deepcopy(gm)
            for name, param in new_gm.named_parameters():
                param.data.zero_()

            hash_val = cast(str, FxGraphCachePickler.get_hash(new_gm))

        return hash_val

    @staticmethod
    def pack(
        serialized_engine: bytes,
        input_names: List[str],
        output_names: List[str],
        weight_name_map: Optional[Dict[Any, Any]],
    ) -> bytes:
        """Pack serialized engine, input names, output names, and weight map into a single blob

        Args:
            serialized_engine (bytes): serialized TRT engine
            input_names (List[str]): input names of TRT engine
            output_names (List[str]): output names of TRT engine
            weight_name_map (Optional[Dict[Any, Any]]): weight name map for refitting

        Returns:
            bytes: packed blob
        """
        return pickle.dumps(
            {
                "serialized_engine": bytes(serialized_engine),
                "input_names": input_names,
                "output_names": output_names,
                "weight_name_map": weight_name_map,
            }
        )

    @staticmethod
    def unpack(
        packed_obj: bytes,
    ) -> Tuple[bytes, List[str], List[str], Optional[Dict[Any, Any]]]:
        """Unpack packed blob into serialized engine, input names, output names, and weight map

        Args:
            packed_obj (bytes): packed blob

        Returns:
            Tuple[bytes, List[str], List[str], Optional[Dict[str, Any]]]: serialized engine, input names, output names, weight name map
        """
        unpacked = pickle.loads(packed_obj)
        return (
            unpacked["serialized_engine"],
            unpacked["input_names"],
            unpacked["output_names"],
            unpacked["weight_name_map"],
        )

    @abstractmethod
    def save(self, hash: str, blob: bytes, *args: Any, **kwargs: Any) -> None:
        """Store blob in cache

        Args:
            hash (str): hash value of the GraphModule
            blob (bytes): packed blob
        """
        pass

    @abstractmethod
    def load(self, hash: str, *args: Any, **kwargs: Any) -> Optional[bytes]:
        """Load blob from storage

        Args:
            hash (str): hash value of the GraphModule

        Returns:
            Optional[bytes]: blob or None if doesn't hit
        """
        pass


class DiskEngineCache(BaseEngineCache):
    dir2hash2size_map: Dict[str, Dict[str, int]] = (
        {}
    )  # dir2hash2size_map["engine_cache_dir"]["hash"] = size

    def __init__(
        self,
        engine_cache_dir: str,
        engine_cache_size: int,
    ) -> None:

        def get_dir_size(path: str) -> int:
            total = 0
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += get_dir_size(entry.path)
            return total

        if not os.path.exists(engine_cache_dir):
            os.makedirs(engine_cache_dir, exist_ok=True)
        self.engine_cache_dir = engine_cache_dir
        self.total_engine_cache_size = engine_cache_size
        self.available_engine_cache_size = engine_cache_size - get_dir_size(
            engine_cache_dir
        )
        if engine_cache_dir not in DiskEngineCache.dir2hash2size_map:
            DiskEngineCache.dir2hash2size_map[engine_cache_dir] = {}

    def has_available_cache_size(self, needed_size: int) -> bool:
        """Check if the cache has available space for saving object

        Args:
            needed_size (int): needed size for saving object

        Returns:
            bool: whether the cache has available size for saving object
        """
        return needed_size <= self.available_engine_cache_size

    def clear_cache(self, needed_min_size: int) -> None:
        """Clear the cache to make sure at least `needed_min_size` bytes are available, if possible

        Args:
            needed_min_size (int): the minimum needed size
        """

        def LRU() -> None:
            """Clear the Least Recently Used engine in the cache"""
            # Get the list of engine directories
            engines_hash_values = os.listdir(self.engine_cache_dir)
            # Sort the engine directories by modification time (oldest first)
            engines_hash_values.sort(
                key=lambda x: os.path.getmtime(os.path.join(self.engine_cache_dir, x))
            )
            # Iterate over the engine directories and remove the oldest ones until enough space is available
            for engine_hash in engines_hash_values:
                if self.available_engine_cache_size >= needed_min_size:
                    break
                engine_path = os.path.join(self.engine_cache_dir, engine_hash)
                try:
                    # Remove the entire directory
                    shutil.rmtree(engine_path)
                    # Update the available cache size
                    self.available_engine_cache_size += (
                        DiskEngineCache.dir2hash2size_map[self.engine_cache_dir].pop(
                            engine_hash, 0
                        )
                    )
                    _LOGGER.info(
                        f"Removed the engine cache at {engine_path}, available cache size: {self.available_engine_cache_size} bytes."
                    )
                except Exception as e:
                    _LOGGER.warning(
                        f"Failed to clear the engine cache at {engine_path}: {e}"
                    )

        if needed_min_size > self.total_engine_cache_size:
            _LOGGER.warning(
                f"The needed minimum size {needed_min_size} is larger than the total cache size {self.total_engine_cache_size}. Nothing will be cleared."
            )
        else:
            LRU()

    def save(
        self,
        hash: str,
        blob: bytes,
    ) -> None:
        blob_size = len(blob)
        if blob_size > self.total_engine_cache_size:
            _LOGGER.warning(
                f"The serialized engine cannot be saved because the size {blob_size} is larger than the total cache size {self.total_engine_cache_size}."
            )
            return

        if not self.has_available_cache_size(blob_size):
            self.clear_cache(blob_size)

        if self.has_available_cache_size(blob_size):
            DiskEngineCache.dir2hash2size_map[self.engine_cache_dir][hash] = blob_size
            self.available_engine_cache_size -= blob_size
            directory = os.path.join(self.engine_cache_dir, hash)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            blob_path = os.path.join(
                directory,
                "blob.bin",
            )
            try:
                with open(blob_path, "wb") as f:
                    f.write(blob)
                _LOGGER.info(f"The blob was saved to {blob_path}")
            except Exception as e:
                del DiskEngineCache.dir2hash2size_map[self.engine_cache_dir][hash]
                self.available_engine_cache_size += blob_size
                shutil.rmtree(directory)
                _LOGGER.warning(f"Failed to save the blob to {blob_path}: {e}")

        else:
            _LOGGER.warning(
                f"The size {blob_size} is still larger than the available cache size {self.available_engine_cache_size}."
            )

    def load(self, hash: str) -> Optional[bytes]:
        directory = os.path.join(self.engine_cache_dir, hash)
        if os.path.exists(directory):
            blob_path = os.path.join(directory, "blob.bin")
            if os.path.exists(blob_path):
                with open(blob_path, "rb") as f:
                    blob = f.read()
                return blob
        return None
