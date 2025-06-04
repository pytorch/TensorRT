import tempfile
from dataclasses import dataclass


@dataclass
class DebuggerConfig:
    log_level: str = "debug"
    save_engine_profile: bool = False
    engine_builder_monitor: bool = True
    logging_dir: str = tempfile.gettempdir()
