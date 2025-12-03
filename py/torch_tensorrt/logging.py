import logging
from typing import Any

import tensorrt as trt
import torch
from torch_tensorrt._features import (
    has_torch_tensorrt_runtime,
    has_torchscript_frontend,
)

logging.captureWarnings(True)
_LOGGER = logging.getLogger("torch_tensorrt [TensorRT Conversion Context]")


class _TRTLogger(trt.ILogger):  # type: ignore[misc]
    def __init__(self) -> None:
        trt.ILogger.__init__(self)

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        # TODO: Move to match once py39 reaches EoL
        if severity == trt.ILogger.Severity.INTERNAL_ERROR:
            _LOGGER.critical(msg)
            raise RuntimeError(msg)
        elif severity == trt.ILogger.Severity.ERROR:
            _LOGGER.error(msg)
        elif severity == trt.ILogger.Severity.WARNING:
            _LOGGER.warning(msg)
        elif severity == trt.ILogger.Severity.INFO:
            _LOGGER.info(msg)
        elif severity == trt.ILogger.Severity.VERBOSE:
            _LOGGER.debug(msg)


TRT_LOGGER = _TRTLogger()


def set_level(level: int, logger: Any = None) -> None:
    """Set log level for both Python and C++ torch_tensorrt loggers.

    Permanently sets the log level until changed again or process exits.
    Automatically handles runtime availability checks.

    This sets the log level for:
    - Specified Python logger (or root torch_tensorrt logger if None)
    - TorchScript frontend C++ logger (if available)
    - Dynamo runtime C++ logger (if available)

    Args:
        level: Python logging level (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
        logger: Optional logger to set level for. If None, sets the root torch_tensorrt logger.

    Example:

        # Set debug logging for entire session
        torch_tensorrt.logging.set_level(logging.DEBUG)

        # Or set for a specific logger
        my_logger = logging.getLogger("torch_tensorrt.dynamo")
        torch_tensorrt.logging.set_level(logging.DEBUG, logger=my_logger)
    """
    # Set the specified logger or default to root torch_tensorrt logger
    if logger is None:
        logging.getLogger("torch_tensorrt").setLevel(level)
        _LOGGER.setLevel(level)
    else:
        logger.setLevel(level)

    if has_torchscript_frontend():
        from torch_tensorrt.ts import logging as ts_logging

        if level == logging.CRITICAL:
            ts_logging.set_reportable_log_level(ts_logging.Level.InternalError)
        elif level == logging.ERROR:
            ts_logging.set_reportable_log_level(ts_logging.Level.Error)
        elif level == logging.WARNING:
            ts_logging.set_reportable_log_level(ts_logging.Level.Warning)
        elif level == logging.INFO:
            ts_logging.set_reportable_log_level(ts_logging.Level.Info)
        elif level == logging.DEBUG:
            ts_logging.set_reportable_log_level(ts_logging.Level.Debug)
        elif level == logging.NOTSET:
            ts_logging.set_reportable_log_level(ts_logging.Level.Graph)
        else:
            raise ValueError(
                f"Invalid log level: {level}. Must be one of: "
                f"logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL, logging.NOTSET"
            )

    elif has_torch_tensorrt_runtime():
        if level == logging.CRITICAL:
            torch.ops.tensorrt.set_logging_level(
                int(trt.ILogger.Severity.INTERNAL_ERROR)
            )
        elif level == logging.ERROR:
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.ERROR))
        elif level == logging.WARNING:
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.WARNING))
        elif level == logging.INFO:
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.INFO))
        elif level == logging.DEBUG:
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.VERBOSE))
        elif level == logging.NOTSET:
            # Graph level (most verbose)
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.VERBOSE) + 1)
        else:
            raise ValueError(
                f"Invalid log level: {level}. Must be one of: "
                f"logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL, logging.NOTSET"
            )


class internal_errors:
    """Context-manager to limit displayed log messages to just internal errors

    Example:

        .. code-block:: py

            with torch_tensorrt.logging.internal_errors():
                outputs = model_torchtrt(inputs)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.CRITICAL)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.InternalError)

        elif has_torch_tensorrt_runtime():
            self.rt_level = torch.ops.tensorrt.get_logging_level()
            torch.ops.tensorrt.set_logging_level(
                int(trt.ILogger.Severity.INTERNAL_ERROR)
            )

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)

        elif has_torch_tensorrt_runtime():
            torch.ops.tensorrt.set_logging_level(self.rt_level)


class errors:
    """Context-manager to limit displayed log messages to just errors and above

    Example:

        .. code-block:: py

            with torch_tensorrt.logging.errors():
                outputs = model_torchtrt(inputs)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.ERROR)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Error)

        elif has_torch_tensorrt_runtime():
            self.rt_level = torch.ops.tensorrt.get_logging_level()
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.ERROR))

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)

        elif has_torch_tensorrt_runtime():
            torch.ops.tensorrt.set_logging_level(self.rt_level)


class warnings:
    """Context-manager to limit displayed log messages to just warnings and above

    Example:

        .. code-block:: py

            with torch_tensorrt.logging.warnings():
                model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.WARNING)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Warning)

        elif has_torch_tensorrt_runtime():
            self.rt_level = torch.ops.tensorrt.get_logging_level()
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.WARNING))

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)

        elif has_torch_tensorrt_runtime():
            torch.ops.tensorrt.set_logging_level(self.rt_level)


class info:
    """Context-manager to display all info and greater severity messages

    Example:

        .. code-block:: py

            with torch_tensorrt.logging.info():
                model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.INFO)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Info)

        elif has_torch_tensorrt_runtime():
            self.rt_level = torch.ops.tensorrt.get_logging_level()
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.INFO))

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)

        elif has_torch_tensorrt_runtime():
            torch.ops.tensorrt.set_logging_level(self.rt_level)


class debug:
    """Context-manager to display full debug information through the logger

    Example:

        .. code-block:: py

            with torch_tensorrt.logging.debug():
                model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.DEBUG)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Debug)

        elif has_torch_tensorrt_runtime():
            self.rt_level = torch.ops.tensorrt.get_logging_level()
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.VERBOSE))

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)

        elif has_torch_tensorrt_runtime():
            torch.ops.tensorrt.set_logging_level(self.rt_level)


class graphs:
    """Context-manager to display the results of intermediate lowering passes
    as well as full debug information through the logger

    Example:

        .. code-block:: py

            with torch_tensorrt.logging.graphs():
                model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.NOTSET)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Graph)

        elif has_torch_tensorrt_runtime():
            self.rt_level = torch.ops.tensorrt.get_logging_level()
            torch.ops.tensorrt.set_logging_level(int(trt.ILogger.Severity.VERBOSE) + 1)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if has_torchscript_frontend():
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)

        elif has_torch_tensorrt_runtime():
            torch.ops.tensorrt.set_logging_level(self.rt_level)
