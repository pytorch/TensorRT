from typing import Any, List, Optional, Dict

import logging
from dataclasses import dataclass, field

@dataclass
class DebuggerConfig:
    log_level: int = logging.getLevelName('DEBUG')
    capture_fx_graph_before: List[str] = field(default_factory=lambda: [])
    capture_fx_graph_after: List[str] = field(default_factory=lambda: [])
    save_engine_profile: bool = False
    engine_builder_monitor: bool = True
    break_in_remove_assert_nodes: bool = True
