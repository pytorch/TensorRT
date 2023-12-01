import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

import tensorrt as trt
import torch

__all__ = [
    "get_cache_mode_batch",
    "get_batch_size",
    "get_batch",
    "read_calibration_cache",
    "write_calibration_cache",
    "CalibrationAlgo",
    "DataLoaderCalibrator",
    "CacheCalibrator",
]


def get_cache_mode_batch(self: object) -> None:
    return None


def get_batch_size(self: object) -> int:
    return 1


def get_batch(self: object, _: Any) -> Optional[List[int]]:
    if self.current_batch_idx + self.batch_size > len(self.data_loader.dataset):
        return None

    batch = next(self.dataset_iterator)
    self.current_batch_idx += self.batch_size
    inputs_gpu = []
    if isinstance(batch, list):
        for example in batch:
            inputs_gpu.append(example.to(self.device).data_ptr())
    else:
        inputs_gpu.append(batch.to(self.device).data_ptr())
    return inputs_gpu


def read_calibration_cache(self: object) -> bytes:
    if self.cache_file and self.use_cache:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                b: bytes = f.read()
                return b
        else:
            raise FileNotFoundError(self.cache_file)
    else:
        return b""


def write_calibration_cache(self: object, cache: bytes) -> None:
    if self.cache_file:
        with open(self.cache_file, "wb") as f:
            f.write(cache)
    else:
        return


class CalibrationAlgo(Enum):
    ENTROPY_CALIBRATION = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
    ENTROPY_CALIBRATION_2 = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    LEGACY_CALIBRATION = trt.CalibrationAlgoType.LEGACY_CALIBRATION
    MINMAX_CALIBRATION = trt.CalibrationAlgoType.MINMAX_CALIBRATION


@dataclass
class DataLoaderCalibrator:
    """
    Constructs a calibrator class in TensorRT and uses pytorch dataloader to load/preproces
    data which is passed during calibration.
    Args:
        dataloader: an instance of pytorch dataloader which iterates through a given dataset.
        algo_type: choice of calibration algorithm.
        cache_file: path to cache file.
        use_cache: flag which enables usage of pre-existing cache.
        device: device on which calibration data is copied to.
    """

    dataloader: torch.utils.data.DataLoader
    algo_type: CalibrationAlgo = CalibrationAlgo.ENTROPY_CALIBRATION_2
    cache_file: str = ""
    use_cache: bool = False
    device: torch.device = torch.device("cuda:0")


@dataclass
class CacheCalibrator(object):
    """
    Constructs a calibrator class in TensorRT which directly uses pre-existing cache file for calibration.
    Args:
        cache_file: path to cache file.
        algo_type: choice of calibration algorithm.
    """

    algo_type: CalibrationAlgo = CalibrationAlgo.ENTROPY_CALIBRATION_2
    cache_file: str = ""
