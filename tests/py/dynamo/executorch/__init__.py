from pkgutil import extend_path

# CI runs these tests from tests/py/dynamo, where this package would otherwise
# shadow the installed ExecuTorch package and hide executorch.exir.
__path__ = extend_path(__path__, __name__)
