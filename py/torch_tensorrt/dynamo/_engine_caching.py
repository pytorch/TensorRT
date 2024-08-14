import copy
import logging
import os
import pickle
import shutil
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch._inductor.codecache import FxGraphCachePickler
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

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
        with maybe_disable_fake_tensor_mode():
            new_gm = copy.deepcopy(gm)
            for name, param in new_gm.named_parameters():
                param.data.zero_()

            hash_val = cast(str, FxGraphCachePickler.get_hash(new_gm))

        return hash_val

    @abstractmethod
    def save(
        self,
        hash: str,
        serialized_engine: bytes,
        input_names: List[str],
        output_names: List[str],
        weight_name_map: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save the serialized engine to hard disk

        Args:
            hash (str): hash value of the GraphModule
            serialized_engine (bytes): serialized TRT engine
            input_names (List[str]): input names of TRT engine
            output_names (List[str]): output names of TRT engine
            weight_name_map (Optional[Dict[str, Any]]): weight name map for refitting

        Returns:
            bool: whether the serialized engine is saved successfully
        """
        pass

    @abstractmethod
    def load(
        self, hash: str
    ) -> Tuple[Optional[bytes], List[str], List[str], Optional[Dict[str, Any]]]:
        """Load the serialized engine from hard disk

        Args:
            hash (str): hash value of the GraphModule

        Returns:
            Sequence[Optional[bytes], List[str], List[str], Optional[Dict[str, Any]]]: serialized engine, input names, output names, weight name map
        """
        pass


class EngineCache(BaseEngineCache):

    def __init__(
        self,
        engine_cache_size: int,
        engine_cache_dir: str,
    ) -> None:
        self.total_engine_cache_size = engine_cache_size
        self.available_engine_cache_size = engine_cache_size
        self.engine_cache_dir = engine_cache_dir
        self.hash2size_map: Dict[str, int] = {}

    def has_available_cache_size(self, needed_size: int) -> bool:
        """Check if the cache has available space for saving the serialized engine

        Args:
            needed_size (int): needed size for erialized TRT engine and/or weight_name_map

        Returns:
            bool: whether the cache has available size for the serialized engine
        """
        return needed_size <= self.available_engine_cache_size

    def clear_cache(self, needed_min_size: int) -> bool:
        """Clear the cache to make sure at least `needed_min_size` bytes are available, if possible

        Args:
            needed_min_size (int): the minimum needed size

        Returns:
            bool: whether the cache is cleared successfully
        """

        def LRU() -> bool:
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
                    self.available_engine_cache_size += self.hash2size_map.pop(
                        engine_hash, 0
                    )
                    _LOGGER.info(
                        f"Removed the engine cache at {engine_path}, available cache size: {self.available_engine_cache_size} bytes."
                    )
                except Exception as e:
                    _LOGGER.warning(
                        f"Failed to clear the engine cache at {engine_path}: {e}"
                    )
                    return False
            return True

        if not os.path.exists(self.engine_cache_dir):
            return False

        _LOGGER.info(
            f"Total cache size: {self.total_engine_cache_size} bytes; available cache size: {self.available_engine_cache_size} bytes. Clearing the cache to make sure at least {needed_min_size} bytes are available."
        )
        return LRU()

    def save(
        self,
        hash: str,
        serialized_engine: bytes,
        input_names: List[str],
        output_names: List[str],
        weight_name_map: Optional[Dict[str, Any]] = None,
    ) -> bool:
        serialized_engine_size = int(serialized_engine.nbytes)
        if weight_name_map is not None:
            serialized_engine_size += sum(
                sys.getsizeof(v) for v in weight_name_map.values()
            )
        if serialized_engine_size > self.total_engine_cache_size:
            _LOGGER.warning(
                f"The serialized engine cannot be saved because the size of the engine {serialized_engine_size} is larger than the total cache size {self.total_engine_cache_size}."
            )
            return False

        # Check if there is enough available cache size for the serialized engine and/or weight_name_map
        if not self.has_available_cache_size(serialized_engine_size):
            self.clear_cache(serialized_engine_size)

        # Save the serialized engine to the cache directory
        if self.has_available_cache_size(serialized_engine_size):
            self.hash2size_map[hash] = serialized_engine_size
            self.available_engine_cache_size -= serialized_engine_size
            directory = os.path.join(self.engine_cache_dir, hash)

            engine_path = os.path.join(
                directory,
                "engine.trt",
            )
            io_names_path = os.path.join(
                directory,
                "io_names.pkl",
            )
            try:
                os.makedirs(os.path.dirname(engine_path), exist_ok=True)
                with open(engine_path, "wb") as f:
                    f.write(serialized_engine)
                os.makedirs(os.path.dirname(io_names_path), exist_ok=True)
                with open(io_names_path, "wb") as f:
                    pickle.dump(
                        {"input_names": input_names, "output_names": output_names}, f
                    )
                _LOGGER.info(f"The TRT engine was saved to {engine_path}")
            except Exception as e:
                del self.hash2size_map[hash]
                self.available_engine_cache_size += serialized_engine_size
                shutil.rmtree(directory)
                _LOGGER.warning(f"Failed to save the TRT engine to {engine_path}: {e}")
                return False

            if weight_name_map is not None:
                weight_name_map_path = os.path.join(
                    directory,
                    "weight_name_map.pkl",
                )
                try:
                    os.makedirs(os.path.dirname(weight_name_map_path), exist_ok=True)
                    with open(weight_name_map_path, "wb") as f:
                        pickle.dump(weight_name_map, f)
                    _LOGGER.info(
                        f"The weight_name_map was saved to {weight_name_map_path}"
                    )
                except Exception as e:
                    del self.hash2size_map[hash]
                    self.available_engine_cache_size += serialized_engine_size
                    shutil.rmtree(directory)
                    _LOGGER.warning(
                        f"Failed to save the weight_name_map to {weight_name_map_path}: {e}"
                    )
                    return False

            return True

        else:
            _LOGGER.warning(
                f"The serialized engine {serialized_engine_size} is still larger than the available cache size {self.available_engine_cache_size}."
            )
            return False

    def load(
        self, hash: str
    ) -> Tuple[Optional[bytes], List[str], List[str], Optional[Dict[str, Any]]]:
        directory = os.path.join(self.engine_cache_dir, hash)
        if os.path.exists(directory):
            # load engine
            serialized_engine = None
            engine_path = os.path.join(directory, "engine.trt")
            if os.path.exists(engine_path):
                with open(engine_path, "rb") as f:
                    serialized_engine = f.read()

            input_names = []
            output_names = []
            io_names_path = os.path.join(directory, "io_names.pkl")
            if os.path.exists(io_names_path):
                with open(io_names_path, "rb") as f:
                    io_names = pickle.load(f)
                    input_names = io_names["input_names"]
                    output_names = io_names["output_names"]

            # load weight_name_map
            weight_name_map = None
            weight_name_map_path = os.path.join(directory, "weight_name_map.pkl")
            if os.path.exists(weight_name_map_path):
                with open(weight_name_map_path, "rb") as f:
                    weight_name_map = pickle.load(f)
            return serialized_engine, input_names, output_names, weight_name_map
        else:
            return None, [], [], {}
