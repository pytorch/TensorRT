from enum import Enum
from typing import Any

from torch_tensorrt._C import (
    LogLevel,
    _get_is_colored_output_on,
    _get_logging_prefix,
    _get_reportable_log_level,
    _log,
    _set_is_colored_output_on,
    _set_logging_prefix,
    _set_reportable_log_level,
)


class Level(Enum):
    """Enum to set the minimum required logging level to print a message to stdout"""

    InternalError = LogLevel.INTERNAL_ERROR
    Error = LogLevel.ERROR
    Warning = LogLevel.WARNING
    Info = LogLevel.INFO
    Debug = LogLevel.DEBUG
    Graph = LogLevel.GRAPH

    @staticmethod
    def _to_internal_level(external: "Level") -> LogLevel:
        if external == Level.InternalError:
            return LogLevel.INTERNAL_ERROR
        elif external == Level.Error:
            return LogLevel.ERROR
        elif external == Level.Warning:
            return LogLevel.WARNING
        elif external == Level.Info:
            return LogLevel.INFO
        elif external == Level.Debug:
            return LogLevel.DEBUG
        elif external == Level.Graph:
            return LogLevel.GRAPH
        else:
            raise ValueError("Unknown log severity")


def get_logging_prefix() -> str:
    """Get the prefix set for logging messages

    Returns:
        str: Prefix used for logger
    """
    return str(_get_logging_prefix())


def set_logging_prefix(prefix: str) -> None:
    """Set the prefix used when logging messages

    Args:
        prefix (str): Prefix to use for logging messages
    """
    _set_logging_prefix(prefix)


def get_reportable_log_level() -> Level:
    """Get the level required for a message to be printed in the log

    Returns:
        torch_tensorrt.logging.Level: The enum representing the level required to print
    """
    return Level(_get_reportable_log_level())


def set_reportable_log_level(level: Level) -> None:
    """Set the level required for a message to be printed to the log

    Args:
        level (torch_tensorrt.logging.Level): The enum representing the level required to print
    """
    _set_reportable_log_level(Level._to_internal_level(level))


def get_is_colored_output_on() -> bool:
    """Get if colored output is enabled for logging

    Returns:
        bool: If colored output is one
    """
    return bool(_get_is_colored_output_on())


def set_is_colored_output_on(colored_output_on: bool) -> None:
    """Enable or disable color in the log output

    Args:
        colored_output_on (bool): If colored output should be enabled or not
    """
    _set_is_colored_output_on(colored_output_on)


def log(level: Level, msg: str) -> None:
    """Add a new message to the log

    Adds a new message to the log at a specified level. The message
    will only get printed out if Level > reportable_log_level

    Args:
        level (torch_tensorrt.logging.Level): Severity of the message
        msg (str): Actual message text
    """
    _log(Level._to_internal_level(level), msg)

    InternalError = LogLevel.INTERNAL_ERROR
    Error = LogLevel.ERROR
    Warning = LogLevel.WARNING
    Info = LogLevel.INFO
    Debug = LogLevel.DEBUG
    Graph = LogLevel.GRAPH


class internal_errors:
    """Context-manager to limit displayed log messages to just internal errors

    Example::

    with torch_tensorrt.logging.internal_errors():
        outputs = model_torchtrt(inputs)
    """

    def __enter__(self) -> None:
        self.external_lvl = get_reportable_log_level()
        set_reportable_log_level(Level.InternalError)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_reportable_log_level(self.external_lvl)


class errors:
    """Context-manager to limit displayed log messages to just errors and above

    Example::

    with torch_tensorrt.logging.errors():
        outputs = model_torchtrt(inputs)
    """

    def __enter__(self) -> None:
        self.external_lvl = get_reportable_log_level()
        set_reportable_log_level(Level.Error)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_reportable_log_level(self.external_lvl)


class warnings:
    """Context-manager to limit displayed log messages to just warnings and above

    Example::

    with torch_tensorrt.logging.warnings():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = get_reportable_log_level()
        set_reportable_log_level(Level.Warning)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_reportable_log_level(self.external_lvl)


class info:
    """Context-manager to display all info and greater severity messages

    Example::

    with torch_tensorrt.logging.info():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = get_reportable_log_level()
        set_reportable_log_level(Level.Info)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_reportable_log_level(self.external_lvl)


class debug:
    """Context-manager to display full debug information through the logger

    Example::

    with torch_tensorrt.logging.debug():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = get_reportable_log_level()
        set_reportable_log_level(Level.Debug)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_reportable_log_level(self.external_lvl)


class graphs:
    """Context-manager to display the results of intermediate lowering passes
    as well as full debug information through the logger

    Example::

    with torch_tensorrt.logging.graphs():
        model_trt = torch_tensorrt.compile(model, **spec)
    """

    def __enter__(self) -> None:
        self.external_lvl = get_reportable_log_level()
        set_reportable_log_level(Level.Graph)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_reportable_log_level(self.external_lvl)
