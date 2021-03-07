from typing import List, Dict, Any
import torch
import os

import trtorch._C
from trtorch._compile_spec import _parse_compile_spec
from trtorch._version import __version__
from types import FunctionType
import tensorrt as trt

def get_cache_mode_batch(self):
    return None

def get_batch_size(self):
    return self.batch_size

def get_batch(self, names):
    if self.current_batch_idx + self.batch_size > self.data_loader.dataset.data.shape[0]:
        return None
    batch = self.dataset_iterator.next()
    self.current_batch_idx += self.batch_size
    print("Calibrating batch: ", self.current_batch_idx)
    # Treat the first element as input and others as targets.
    if isinstance(batch, list):
        batch = batch[0].to(torch.device('cuda:0'))
    return [batch.data_ptr()]

def read_calibration_cache(self):
    if self.use_cache:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

def write_calibration_cache(self, cache):
    with open(self.cache_file, "wb") as f:
        f.write(cache)

def make_int8_calibrator(dataloader, cache_file, use_cache, algo_type):
    # Define attributes and member functions for the calibrator class
    attribute_mapping={'data_loader' : dataloader,
                       'current_batch_idx' : 0,
                       'batch_size' : dataloader.batch_size,
                       'dataset_iterator' : iter(dataloader),
                       'cache_file' : cache_file,
                       'use_cache' : use_cache,
                       'get_batch_size' : get_batch_size,
                       'get_batch': get_cache_mode_batch if use_cache else get_batch,
                       'read_calibration_cache' : read_calibration_cache,
                       'write_calibration_cache' : write_calibration_cache}

    # Using type metaclass to construct calibrator class based on algorithm type
    if algo_type == trtorch._C.CalibrationAlgo.ENTROPY_CALIBRATION:
        return type('DataLoaderCalibrator', (trtorch._C.IInt8EntropyCalibrator,), attribute_mapping)()
    elif algo_type == trtorch._C.CalibrationAlgo.ENTROPY_CALIBRATION_2:
        return type('DataLoaderCalibrator', (trtorch._C.IInt8MinMaxCalibrator,), attribute_mapping)()
    elif algo_type == trtorch._C.CalibrationAlgo.LEGACY_CALIBRATION:
        return type('DataLoaderCalibrator', (trtorch._C.IInt8LegacyCalibrator,), attribute_mapping)()
    elif algo_type == trtorch._C.CalibrationAlgo.MINMAX_CALIBRATION:
        return type('DataLoaderCalibrator', (trtorch._C.IInt8MinMaxCalibrator,), attribute_mapping)()
    else:
        return ValueError("Invalid calibration algorithm type. Please select among ENTROPY_CALIBRATION, ENTROPY_CALIBRATION, LEGACY_CALIBRATION or MINMAX_CALIBRATION");
