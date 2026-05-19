from pkgutil import extend_path

# Tests are run from tests/py/dynamo, where this directory is importable as the
# top-level "executorch" package. Keep the real installed ExecuTorch namespace
# visible so imports like executorch.exir resolve correctly.
__path__ = extend_path(__path__, __name__)
