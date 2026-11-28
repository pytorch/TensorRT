import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def policy(tmp_path):
    from torch_tensorrt_mcp.policy import Policy

    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    workspace.mkdir()
    output.mkdir()
    (workspace / "factories.py").write_text(
        "def model(**kwargs): return None\ndef inputs(**kwargs): return ()\n"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "workspace_root": str(workspace),
                "output_root": str(output),
                "models": {
                    "model": {
                        "path": "factories.py",
                        "function": "model",
                        "parameters": ["mode"],
                    }
                },
                "inputs": {
                    "inputs": {
                        "path": "factories.py",
                        "function": "inputs",
                        "parameters": [],
                    }
                },
            }
        )
    )
    return Policy.load(config)
