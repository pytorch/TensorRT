import yaml

__version__: str = "0.0.0"
__cuda_version__: str = "0.0"
__cudnn_version__: str = "0.0"
__tensorrt_version__: str = "0.0"


def load_version_info():
    global __version__
    global __cuda_version__
    global __cudnn_version__
    global __tensorrt_version__
    with open("versions.yml", "r") as stream:
        versions = yaml.safe_load(stream)
        __version__ = versions["__version__"]
        __cuda_version__ = versions["__cuda_version__"]
        __cudnn_version__ = versions["__cudnn_version__"]
        __tensorrt_version__ = versions["__tensorrt_version__"]


load_version_info()


def torch_tensorrt_version():
    print(__version__)


def cuda_version():
    print(__cuda_version__)


def cudnn_version():
    print(__cudnn_version__)


def tensorrt_version():
    print(__tensorrt_version__)
