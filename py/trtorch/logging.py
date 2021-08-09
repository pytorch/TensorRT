from enum import Enum
from trtorch._C import _get_logging_prefix, _set_logging_prefix, \
                        _get_reportable_log_level, _set_reportable_log_level, \
                        _get_is_colored_output_on,  _set_is_colored_output_on, \
                        _log, LogLevel


class Level(Enum):
    """Enum to set the minimum required logging level to print a message to stdout
    """
    InternalError = LogLevel.INTERNAL_ERROR
    Error = LogLevel.ERROR
    Warning = LogLevel.WARNING
    Info = LogLevel.INFO
    Debug = LogLevel.DEBUG
    Graph = LogLevel.GRAPH

    @staticmethod
    def _to_internal_level(external) -> LogLevel:
        if external == Level.InternalError:
            return LogLevel.INTERNAL_ERROR
        if external == Level.Error:
            return LogLevel.ERROR
        if external == Level.Warning:
            return LogLevel.WARNING
        if external == Level.Info:
            return LogLevel.INFO
        if external == Level.Debug:
            return LogLevel.DEBUG
        if external == Level.Graph:
            return LogLevel.GRAPH


def get_logging_prefix() -> str:
    """Get the prefix set for logging messages

    Returns:
        str: Prefix used for logger
    """
    return _get_logging_prefix()


def set_logging_prefix(prefix: str):
    """Set the prefix used when logging messages

    Args:
        prefix (str): Prefix to use for logging messages
    """
    _set_logging_prefix(prefix)


def get_reportable_log_level() -> Level:
    """Get the level required for a message to be printed in the log

    Returns:
        trtorch.logging.Level: The enum representing the level required to print
    """
    return Level(_get_reportable_log_level())


def set_reportable_log_level(level: Level):
    """Set the level required for a message to be printed to the log

    Args:
        level (trtorch.logging.Level): The enum representing the level required to print
    """
    _set_reportable_log_level(Level._to_internal_level(level))


def get_is_colored_output_on() -> bool:
    """Get if colored output is enabled for logging

    Returns:
        bool: If colored output is one
    """
    return _get_is_colored_output_on()


def set_is_colored_output_on(colored_output_on: bool):
    """Enable or disable color in the log output

    Args:
        colored_output_on (bool): If colored output should be enabled or not
    """
    _set_is_colored_output_on(colored_output_on)


def log(level: Level, msg: str):
    """Add a new message to the log

    Adds a new message to the log at a specified level. The message
    will only get printed out if Level > reportable_log_level

    Args:
        level (trtorch.logging.Level): Severity of the message
        msg (str): Actual message text
    """
    _log(Level._to_internal_level(level), msg)
