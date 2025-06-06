import os
import tempfile
from dataclasses import dataclass

DEFAULT_LOGGING_DIR = os.path.join(tempfile.gettempdir(), "torch_tensorrt/debug_logs")


@dataclass
class DebuggerConfig:
    log_level: str = "debug"
    save_engine_profile: bool = False
    engine_builder_monitor: bool = True
    logging_dir: str = DEFAULT_LOGGING_DIR
    profile_format: str = "perfetto"
    save_layer_info: bool = False
