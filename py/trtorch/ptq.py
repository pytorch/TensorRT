from typing import List, Dict, Any
import torch
import os

import trtorch._C
from trtorch._compile_spec import _parse_compile_spec
from trtorch._version import __version__
from trtorch.logging import *
from types import FunctionType
from enum import Enum


class CalibrationAlgo(Enum):
    ENTROPY_CALIBRATION = trtorch._C.CalibrationAlgo.ENTROPY_CALIBRATION
    ENTROPY_CALIBRATION_2 = trtorch._C.CalibrationAlgo.ENTROPY_CALIBRATION_2
    LEGACY_CALIBRATION = trtorch._C.CalibrationAlgo.LEGACY_CALIBRATION
    MINMAX_CALIBRATION = trtorch._C.CalibrationAlgo.MINMAX_CALIBRATION


def get_cache_mode_batch(self):
    return None


def get_batch_size(self):
    return 1


def get_batch(self, names):
    if self.current_batch_idx + self.batch_size > self.data_loader.dataset.data.shape[0]:
        return None

    batch = self.dataset_iterator.next()
    self.current_batch_idx += self.batch_size
    # Treat the first element as input and others as targets.
    if isinstance(batch, list):
        batch = batch[0].to(self.device)
    return [batch.data_ptr()]


def read_calibration_cache(self):
    if self.cache_file and self.use_cache:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()


def write_calibration_cache(self, cache):
    if self.cache_file:
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class DataLoaderCalibrator(object):

    def __init__(self, dataloader, **kwargs):
        self.algo_type = kwargs.get("algo_type", trtorch.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2)
        self.cache_file = kwargs.get("cache_file", None)
        self.use_cache = kwargs.get("use_cache", False)
        self.device = kwargs.get("device", torch.device("cuda:0"))

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            log(Level.Error,
                "Dataloader : {} is not a valid instance of torch.utils.data.DataLoader".format(dataloader))

        if not self.cache_file:
            if self.use_cache:
                log(Level.Debug, "Using existing cache_file {} for calibration".format(self.cache_file))
            else:
                log(Level.Debug, "Overwriting existing calibration cache file.")
        else:
            if self.use_cache:
                log(Level.Error, "Input cache file is None but use_cache is set to True in INT8 mode.")

        # Define attributes and member functions for the calibrator class
        self.attribute_mapping = {
            'data_loader': dataloader,
            'current_batch_idx': 0,
            'batch_size': dataloader.batch_size,
            'dataset_iterator': iter(dataloader),
            'cache_file': self.cache_file,
            'device': self.device,
            'use_cache': self.use_cache,
            'get_batch_size': get_batch_size,
            'get_batch': get_cache_mode_batch if self.use_cache else get_batch,
            'read_calibration_cache': read_calibration_cache,
            'write_calibration_cache': write_calibration_cache
        }

    def __call__(self):
        # Using type metaclass to construct calibrator class based on algorithm type
        if self.algo_type == CalibrationAlgo.ENTROPY_CALIBRATION:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8EntropyCalibrator,), self.attribute_mapping)()
        elif self.algo_type == CalibrationAlgo.ENTROPY_CALIBRATION_2:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8MinMaxCalibrator,), self.attribute_mapping)()
        elif self.algo_type == CalibrationAlgo.LEGACY_CALIBRATION:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8LegacyCalibrator,), self.attribute_mapping)()
        elif self.algo_type == CalibrationAlgo.MINMAX_CALIBRATION:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8MinMaxCalibrator,), self.attribute_mapping)()
        else:
            log(
                Level.Error,
                "Invalid calibration algorithm type. Please select among ENTROPY_CALIBRATION, ENTROPY_CALIBRATION, LEGACY_CALIBRATION or MINMAX_CALIBRATION"
            )


class CacheCalibrator(object):

    def __init__(self, cache_file, **kwargs):
        self.cache_file = cache_file
        self.algo_type = kwargs.get("algo_type", trtorch.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2)

        if os.path.isfile(self.cache_file):
            log(Level.Debug, "Using existing cache_file {} for calibration".format(self.cache_file))
        else:
            log(Level.Error, "Invalid calibration cache file.")

        # Define attributes and member functions for the calibrator class
        self.attribute_mapping = {
            'use_cache': True,
            'cache_file': self.cache_file,
            'get_batch_size': get_batch_size,
            'get_batch': get_cache_mode_batch,
            'read_calibration_cache': read_calibration_cache,
            'write_calibration_cache': write_calibration_cache
        }

    def __call__(self):
        # Using type metaclass to construct calibrator class based on algorithm type
        if self.algo_type == CalibrationAlgo.ENTROPY_CALIBRATION:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8EntropyCalibrator,), self.attribute_mapping)()
        elif self.algo_type == CalibrationAlgo.ENTROPY_CALIBRATION_2:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8MinMaxCalibrator,), self.attribute_mapping)()
        elif self.algo_type == CalibrationAlgo.LEGACY_CALIBRATION:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8LegacyCalibrator,), self.attribute_mapping)()
        elif self.algo_type == CalibrationAlgo.MINMAX_CALIBRATION:
            return type('DataLoaderCalibrator', (trtorch._C.IInt8MinMaxCalibrator,), self.attribute_mapping)()
        else:
            log(
                Level.Error,
                "Invalid calibration algorithm type. Please select among ENTROPY_CALIBRATION, ENTROPY_CALIBRATION, LEGACY_CALIBRATION or MINMAX_CALIBRATION"
            )
