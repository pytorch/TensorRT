from typing import List, Dict, Any
import torch
import os

from torch_tensorrt import _C
from torch_tensorrt._version import __version__
from torch_tensorrt.logging import *
from types import FunctionType
from enum import Enum


class CalibrationAlgo(Enum):
    ENTROPY_CALIBRATION = _C.CalibrationAlgo.ENTROPY_CALIBRATION
    ENTROPY_CALIBRATION_2 = _C.CalibrationAlgo.ENTROPY_CALIBRATION_2
    LEGACY_CALIBRATION = _C.CalibrationAlgo.LEGACY_CALIBRATION
    MINMAX_CALIBRATION = _C.CalibrationAlgo.MINMAX_CALIBRATION


def get_cache_mode_batch(self):
    return None


def get_batch_size(self):
    return 1


def get_batch(self, names):
    if self.current_batch_idx + self.batch_size > len(self.data_loader.dataset):
        return None

    batch = self.dataset_iterator.next()
    self.current_batch_idx += self.batch_size
    inputs_gpu = []
    if isinstance(batch, list):
        for example in batch:
            inputs_gpu.append(example.to(self.device).data_ptr())
    else:
        inputs_gpu.append(batch.to(self.device).data_ptr())
    return inputs_gpu


def read_calibration_cache(self):
    if self.cache_file and self.use_cache:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    else:
        return b""


def write_calibration_cache(self, cache):
    if self.cache_file:
        with open(self.cache_file, "wb") as f:
            f.write(cache)
    else:
        return b""


class DataLoaderCalibrator(object):
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

    def __init__(self, **kwargs):
        pass

    def __new__(cls, *args, **kwargs):
        dataloader = args[0]
        algo_type = kwargs.get("algo_type", CalibrationAlgo.ENTROPY_CALIBRATION_2)
        cache_file = kwargs.get("cache_file", None)
        use_cache = kwargs.get("use_cache", False)
        device = kwargs.get("device", torch.device("cuda:0"))

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            log(
                Level.Error,
                "Dataloader : {} is not a valid instance of torch.utils.data.DataLoader".format(
                    dataloader
                ),
            )

        if not cache_file:
            if use_cache:
                log(
                    Level.Debug,
                    "Using existing cache_file {} for calibration".format(cache_file),
                )
            else:
                log(Level.Debug, "Overwriting existing calibration cache file.")
        else:
            if use_cache:
                log(
                    Level.Error,
                    "Input cache file is None but use_cache is set to True in INT8 mode.",
                )

        # Define attributes and member functions for the calibrator class
        attribute_mapping = {
            "data_loader": dataloader,
            "current_batch_idx": 0,
            "batch_size": dataloader.batch_size,
            "dataset_iterator": iter(dataloader),
            "cache_file": cache_file,
            "device": device,
            "use_cache": use_cache,
            "get_batch_size": get_batch_size,
            "get_batch": get_cache_mode_batch if use_cache else get_batch,
            "read_calibration_cache": read_calibration_cache,
            "write_calibration_cache": write_calibration_cache,
        }

        # Using type metaclass to construct calibrator class based on algorithm type
        if algo_type == CalibrationAlgo.ENTROPY_CALIBRATION:
            return type(
                "DataLoaderCalibrator", (_C.IInt8EntropyCalibrator,), attribute_mapping
            )()
        elif algo_type == CalibrationAlgo.ENTROPY_CALIBRATION_2:
            return type(
                "DataLoaderCalibrator", (_C.IInt8MinMaxCalibrator,), attribute_mapping
            )()
        elif algo_type == CalibrationAlgo.LEGACY_CALIBRATION:
            return type(
                "DataLoaderCalibrator", (_C.IInt8LegacyCalibrator,), attribute_mapping
            )()
        elif algo_type == CalibrationAlgo.MINMAX_CALIBRATION:
            return type(
                "DataLoaderCalibrator", (_C.IInt8MinMaxCalibrator,), attribute_mapping
            )()
        else:
            log(
                Level.Error,
                "Invalid calibration algorithm type. Please select among ENTROPY_CALIBRATION, ENTROPY_CALIBRATION, LEGACY_CALIBRATION or MINMAX_CALIBRATION",
            )


class CacheCalibrator(object):
    """
    Constructs a calibrator class in TensorRT which directly uses pre-existing cache file for calibration.
    Args:
        cache_file: path to cache file.
        algo_type: choice of calibration algorithm.
    """

    def __init__(self, **kwargs):
        pass

    def __new__(cls, *args, **kwargs):
        cache_file = args[0]
        algo_type = kwargs.get("algo_type", CalibrationAlgo.ENTROPY_CALIBRATION_2)

        if os.path.isfile(cache_file):
            log(
                Level.Debug,
                "Using existing cache_file {} for calibration".format(cache_file),
            )
        else:
            log(Level.Error, "Invalid calibration cache file.")

        # Define attributes and member functions for the calibrator class
        attribute_mapping = {
            "use_cache": True,
            "cache_file": cache_file,
            "get_batch_size": get_batch_size,
            "get_batch": get_cache_mode_batch,
            "read_calibration_cache": read_calibration_cache,
            "write_calibration_cache": write_calibration_cache,
        }
        # Using type metaclass to construct calibrator class based on algorithm type
        if algo_type == CalibrationAlgo.ENTROPY_CALIBRATION:
            return type(
                "DataLoaderCalibrator", (_C.IInt8EntropyCalibrator,), attribute_mapping
            )()
        elif algo_type == CalibrationAlgo.ENTROPY_CALIBRATION_2:
            return type(
                "DataLoaderCalibrator", (_C.IInt8MinMaxCalibrator,), attribute_mapping
            )()
        elif algo_type == CalibrationAlgo.LEGACY_CALIBRATION:
            return type(
                "DataLoaderCalibrator", (_C.IInt8LegacyCalibrator,), attribute_mapping
            )()
        elif algo_type == CalibrationAlgo.MINMAX_CALIBRATION:
            return type(
                "DataLoaderCalibrator", (_C.IInt8MinMaxCalibrator,), attribute_mapping
            )()
        else:
            log(
                Level.Error,
                "Invalid calibration algorithm type. Please select among ENTROPY_CALIBRATION, ENTROPY_CALIBRATION, LEGACY_CALIBRATION or MINMAX_CALIBRATION",
            )
