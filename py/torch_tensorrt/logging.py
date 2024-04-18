import logging
from typing import Any

import tensorrt as trt
from torch_tensorrt._features import ENABLED_FEATURES

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


class internal_errors:
    """Context-manager to limit displayed log messages to just internal errors

    Example::

    with torch_tensorrt.logging.internal_errors():
        outputs = model_torchtrt(inputs)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.CRITICAL)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.InternalError)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)


class errors:
    """Context-manager to limit displayed log messages to just errors and above

    Example::

    with torch_tensorrt.logging.errors():
        outputs = model_torchtrt(inputs)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.ERROR)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Error)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)


class warnings:
    """Context-manager to limit displayed log messages to just warnings and above

    Example::

    with torch_tensorrt.logging.warnings():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.WARNING)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Warning)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)


class info:
    """Context-manager to display all info and greater severity messages

    Example::

    with torch_tensorrt.logging.info():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.INFO)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Info)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)


class debug:
    """Context-manager to display full debug information through the logger

    Example::

    with torch_tensorrt.logging.debug():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.DEBUG)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Debug)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)


class graphs:
    """Context-manager to display the results of intermediate lowering passes
    as well as full debug information through the logger

    Example::

    with torch_tensorrt.logging.graphs():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.NOTSET)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            self.ts_level = ts_logging.get_reportable_log_level()
            ts_logging.set_reportable_log_level(ts_logging.Level.Graph)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        _LOGGER.setLevel(self.external_lvl)

        if ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt.ts import logging as ts_logging

            ts_logging.set_reportable_log_level(self.ts_level)
