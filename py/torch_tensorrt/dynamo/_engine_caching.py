import ast
import copy
import logging
import os
import shutil
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
    ) -> bool:
        """Save the serialized engine to hard disk

        Args:
            hash (str): hash value of the GraphModule
            serialized_engine (bytes): serialized TRT engine
            input_names (List[str]): input names of TRT engine
            output_names (List[str]): output names of TRT engine

        Returns:
            bool: whether the serialized engine is saved successfully
        """
        pass

    @abstractmethod
    def load(self, hash: str) -> Tuple[Optional[bytes], List[str], List[str]]:
        """Load the serialized engine from hard disk

        Args:
            hash (str): hash value of the GraphModule

        Returns:
            Sequence[Optional[bytes], List[str], List[str]]: serialized TRT engine, input names of TRT Engine, output names of TRT Engine
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

    def has_available_cache_size(self, serialized_engine: bytes) -> bool:
        """Check if the cache has available space for saving the serialized engine

        Args:
            serialized_engine (bytes): serialized TRT engine

        Returns:
            bool: whether the cache has available size for the serialized engine
        """
        return int(serialized_engine.nbytes) <= self.available_engine_cache_size

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
    ) -> bool:
        serialized_engine_size = int(serialized_engine.nbytes)
        if serialized_engine_size > self.total_engine_cache_size:
            _LOGGER.warning(
                f"The serialized engine cannot be saved because the size of the engine {serialized_engine_size} is larger than the total cache size {self.total_engine_cache_size}."
            )
            return False

        # Check if there is enough available cache size for the serialized engine
        if not self.has_available_cache_size(serialized_engine):
            self.clear_cache(serialized_engine_size)

        # Save the serialized engine to the cache directory
        if self.has_available_cache_size(serialized_engine):
            path = os.path.join(
                self.engine_cache_dir,
                f"{hash}/engine--{input_names}--{output_names}.trt",
            )
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(serialized_engine)
                self.hash2size_map[hash] = serialized_engine_size
                self.available_engine_cache_size -= serialized_engine_size
                _LOGGER.info(f"A TRT engine was cached to {path}")

            except Exception as e:
                _LOGGER.warning(f"Failed to save the TRT engine to {path}: {e}")
                return False

            return True

        else:
            _LOGGER.warning(
                f"The serialized engine {serialized_engine_size} is still larger than the available cache size {self.available_engine_cache_size}."
            )
            return False

    def load(self, hash: str) -> Tuple[Optional[bytes], List[str], List[str]]:
        directory = os.path.join(self.engine_cache_dir, hash)
        if os.path.exists(directory):
            engine_list = os.listdir(directory)
            assert (
                len(engine_list) == 1
            ), f"There are more than one engine {engine_list} under {directory}."
            path = os.path.join(directory, engine_list[0])
            input_names_str, output_names_str = (
                engine_list[0].split(".trt")[0].split("--")[1:]
            )
            input_names = ast.literal_eval(input_names_str)
            output_names = ast.literal_eval(output_names_str)
            with open(path, "rb") as f:
                serialized_engine = f.read()
                return serialized_engine, input_names, output_names
        else:
            return None, [], []
