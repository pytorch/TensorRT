import sys
from typing import Any, List, Optional

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import os
from enum import Enum

import torch
from torch_tensorrt import _C
from torch_tensorrt.logging import Level, log


class CalibrationAlgo(Enum):
    ENTROPY_CALIBRATION = _C.CalibrationAlgo.ENTROPY_CALIBRATION
    ENTROPY_CALIBRATION_2 = _C.CalibrationAlgo.ENTROPY_CALIBRATION_2
    LEGACY_CALIBRATION = _C.CalibrationAlgo.LEGACY_CALIBRATION
    MINMAX_CALIBRATION = _C.CalibrationAlgo.MINMAX_CALIBRATION


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


# deepcopy (which involves pickling) is performed on the compile_spec internally during compilation.
# We register this __reduce__ function for pickler to identity the calibrator object returned by DataLoaderCalibrator during deepcopy.
# This should be the object's local name relative to the module https://docs.python.org/3/library/pickle.html#object.__reduce__
def __reduce__(self: object) -> str:
    return self.__class__.__name__


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

    def __init__(self, **kwargs: Any):
        pass

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
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
            "__reduce__": __reduce__,  # used when you deepcopy the DataLoaderCalibrator object
        }

        # Using type metaclass to construct calibrator class based on algorithm type
        if algo_type == CalibrationAlgo.ENTROPY_CALIBRATION:
            calib_ec: Self = type(
                "Int8EntropyCalibrator", (_C.IInt8EntropyCalibrator,), attribute_mapping
            )()
            return calib_ec
        elif algo_type == CalibrationAlgo.ENTROPY_CALIBRATION_2:
            calib_ec2: Self = type(
                "Int8EntropyCalibrator2",
                (_C.IInt8EntropyCalibrator2,),
                attribute_mapping,
            )()
            return calib_ec2
        elif algo_type == CalibrationAlgo.LEGACY_CALIBRATION:
            calib_lc: Self = type(
                "Int8LegacyCalibrator", (_C.IInt8LegacyCalibrator,), attribute_mapping
            )()
            return calib_lc
        elif algo_type == CalibrationAlgo.MINMAX_CALIBRATION:
            calib_mmc: Self = type(
                "Int8MinMaxCalibrator", (_C.IInt8MinMaxCalibrator,), attribute_mapping
            )()
            return calib_mmc
        else:
            raise ValueError(
                "Invalid calibration algorithm type. Please select among ENTROPY_CALIBRATION, ENTROPY_CALIBRATION, LEGACY_CALIBRATION or MINMAX_CALIBRATION"
            )


class CacheCalibrator(object):
    """
    Constructs a calibrator class in TensorRT which directly uses pre-existing cache file for calibration.
    Args:
        cache_file: path to cache file.
        algo_type: choice of calibration algorithm.
    """

    def __init__(self, **kwargs: Any):
        pass

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
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
            calib_ec: Self = type(
                "DataLoaderCalibrator", (_C.IInt8EntropyCalibrator,), attribute_mapping
            )()
            return calib_ec
        elif algo_type == CalibrationAlgo.ENTROPY_CALIBRATION_2:
            calib_ec2: Self = type(
                "DataLoaderCalibrator", (_C.IInt8MinMaxCalibrator,), attribute_mapping
            )()
            return calib_ec2
        elif algo_type == CalibrationAlgo.LEGACY_CALIBRATION:
            calib_lc: Self = type(
                "DataLoaderCalibrator", (_C.IInt8LegacyCalibrator,), attribute_mapping
            )()
            return calib_lc
        elif algo_type == CalibrationAlgo.MINMAX_CALIBRATION:
            calib_mmc: Self = type(
                "DataLoaderCalibrator", (_C.IInt8MinMaxCalibrator,), attribute_mapping
            )()
            return calib_mmc
        else:
            raise ValueError(
                "Invalid calibration algorithm type. Please select among ENTROPY_CALIBRATION, ENTROPY_CALIBRATION, LEGACY_CALIBRATION or MINMAX_CALIBRATION"
            )
