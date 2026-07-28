import sys
from pathlib import Path

# Allow `from conversion.harness import ...` and other sibling-package imports
# when pytest is invoked from the repo root against this subdirectory.
_dynamo_tests = str(Path(__file__).parent.parent)
if _dynamo_tests not in sys.path:
    sys.path.insert(0, _dynamo_tests)
