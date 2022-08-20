from distutils.command.clean import clean
import nox
import os
import sys

# Use system installed Python packages
PYT_PATH = (
    "/opt/conda/lib/python3.8/site-packages"
    if not "PYT_PATH" in os.environ
    else os.environ["PYT_PATH"]
)
print(f"Using python path {PYT_PATH}")

# Set the root directory to the directory of the noxfile unless the user wants to
# TOP_DIR
TOP_DIR = (
    os.path.dirname(os.path.realpath(__file__))
    if not "TOP_DIR" in os.environ
    else os.environ["TOP_DIR"]
)
print(f"Test root directory {TOP_DIR}")

# Set the USE_CXX11=1 to use cxx11_abi
USE_CXX11 = 0 if not "USE_CXX11" in os.environ else os.environ["USE_CXX11"]
if USE_CXX11:
    print("Using cxx11 abi")

# Set the USE_HOST_DEPS=1 to use host dependencies for tests
USE_HOST_DEPS = 0 if not "USE_HOST_DEPS" in os.environ else os.environ["USE_HOST_DEPS"]
if USE_HOST_DEPS:
    print("Using dependencies from host python")

SUPPORTED_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]

nox.options.sessions = [
    "l0_api_tests-" + "{}.{}".format(sys.version_info.major, sys.version_info.minor)
]


def install_deps(session):
    print("Installing deps")
    session.install("-r", os.path.join(TOP_DIR, "py", "requirements.txt"))
    session.install("-r", os.path.join(TOP_DIR, "tests", "py", "requirements.txt"))


def download_models(session):
    print("Downloading test models")
    session.install("-r", os.path.join(TOP_DIR, "tests", "modules", "requirements.txt"))
    print(TOP_DIR)
    session.chdir(os.path.join(TOP_DIR, "tests", "modules"))
    if USE_HOST_DEPS:
        session.run_always("python", "hub.py", env={"PYTHONPATH": PYT_PATH})
    else:
        session.run_always("python", "hub.py")


def install_torch_trt(session):
    print("Installing latest torch-tensorrt build")
    session.chdir(os.path.join(TOP_DIR, "py"))
    if USE_CXX11:
        session.run("python", "setup.py", "develop", "--use-cxx11-abi")
    else:
        session.run("python", "setup.py", "develop")


def download_datasets(session):
    print(
        "Downloading dataset to path",
        os.path.join(TOP_DIR, "examples/int8/training/vgg16"),
    )
    session.chdir(os.path.join(TOP_DIR, "examples/int8/training/vgg16"))
    session.run_always(
        "wget", "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", external=True
    )
    session.run_always("tar", "-xvzf", "cifar-10-binary.tar.gz", external=True)
    session.run_always(
        "mkdir",
        "-p",
        os.path.join(TOP_DIR, "tests/accuracy/datasets/data"),
        external=True,
    )
    session.run_always(
        "cp",
        "-rpf",
        os.path.join(TOP_DIR, "examples/int8/training/vgg16/cifar-10-batches-bin"),
        os.path.join(TOP_DIR, "tests/accuracy/datasets/data/cidar-10-batches-bin"),
        external=True,
    )


def train_model(session):
    session.chdir(os.path.join(TOP_DIR, "examples/int8/training/vgg16"))
    session.install("-r", "requirements.txt")
    if os.path.exists("vgg16_ckpts/ckpt_epoch25.pth"):
        session.run_always("python", "export_ckpt.py", "vgg16_ckpts/ckpt_epoch25.pth")
        return
    if USE_HOST_DEPS:
        session.run_always(
            "python",
            "main.py",
            "--lr",
            "0.01",
            "--batch-size",
            "128",
            "--drop-ratio",
            "0.15",
            "--ckpt-dir",
            "vgg16_ckpts",
            "--epochs",
            "25",
            env={"PYTHONPATH": PYT_PATH},
        )

        session.run_always(
            "python",
            "export_ckpt.py",
            "vgg16_ckpts/ckpt_epoch25.pth",
            env={"PYTHONPATH": PYT_PATH},
        )
    else:
        session.run_always(
            "python",
            "main.py",
            "--lr",
            "0.01",
            "--batch-size",
            "128",
            "--drop-ratio",
            "0.15",
            "--ckpt-dir",
            "vgg16_ckpts",
            "--epochs",
            "25",
        )

        session.run_always("python", "export_ckpt.py", "vgg16_ckpts/ckpt_epoch25.pth")


def finetune_model(session):
    # Install pytorch-quantization dependency
    session.install(
        "pytorch-quantization", "--extra-index-url", "https://pypi.ngc.nvidia.com"
    )
    session.chdir(os.path.join(TOP_DIR, "examples/int8/training/vgg16"))

    if USE_HOST_DEPS:
        session.run_always(
            "python",
            "finetune_qat.py",
            "--lr",
            "0.01",
            "--batch-size",
            "128",
            "--drop-ratio",
            "0.15",
            "--ckpt-dir",
            "vgg16_ckpts",
            "--start-from",
            "25",
            "--epochs",
            "26",
            env={"PYTHONPATH": PYT_PATH},
        )

        # Export model
        session.run_always(
            "python",
            "export_qat.py",
            "vgg16_ckpts/ckpt_epoch26.pth",
            env={"PYTHONPATH": PYT_PATH},
        )
    else:
        session.run_always(
            "python",
            "finetune_qat.py",
            "--lr",
            "0.01",
            "--batch-size",
            "128",
            "--drop-ratio",
            "0.15",
            "--ckpt-dir",
            "vgg16_ckpts",
            "--start-from",
            "25",
            "--epochs",
            "26",
        )

        # Export model
        session.run_always("python", "export_qat.py", "vgg16_ckpts/ckpt_epoch26.pth")


def cleanup(session):
    target = [
        "examples/int8/training/vgg16/*.jit.pt",
        "examples/int8/training/vgg16/vgg16_ckpts",
        "examples/int8/training/vgg16/cifar-10-*",
        "examples/int8/training/vgg16/data",
        "tests/modules/*.jit.pt",
        "tests/py/*.jit.pt",
    ]

    target = " ".join(x for x in [os.path.join(TOP_DIR, i) for i in target])
    session.run_always("bash", "-c", str("rm -rf ") + target, external=True)


def run_base_tests(session):
    print("Running basic tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py"))
    tests = [
        "api",
        "integrations/test_to_backend_api.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_accuracy_tests(session):
    print("Running accuracy tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py"))
    tests = []
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("python", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("python", test)


def copy_model(session):
    model_files = ["trained_vgg16.jit.pt", "trained_vgg16_qat.jit.pt"]

    for file_name in model_files:
        src_file = os.path.join(
            TOP_DIR, str("examples/int8/training/vgg16/") + file_name
        )
        if os.path.exists(src_file):
            session.run_always(
                "cp",
                "-rpf",
                os.path.join(TOP_DIR, src_file),
                os.path.join(TOP_DIR, str("tests/modules/") + file_name),
                external=True,
            )


def run_int8_accuracy_tests(session):
    print("Running accuracy tests")
    copy_model(session)
    session.chdir(os.path.join(TOP_DIR, "tests/py"))
    tests = [
        "ptq/test_ptq_to_backend.py",
        "ptq/test_ptq_dataloader_calibrator.py",
        "qat/",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_trt_compatibility_tests(session):
    print("Running TensorRT compatibility tests")
    copy_model(session)
    session.chdir(os.path.join(TOP_DIR, "tests/py"))
    tests = [
        "test_trt_intercompatibility.py",
        "test_ptq_trt_calibrator.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_dla_tests(session):
    print("Running DLA tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py"))
    tests = [
        "test_api_dla.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_multi_gpu_tests(session):
    print("Running multi GPU tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py"))
    tests = [
        "test_multi_gpu.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_l0_api_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_models(session)
    run_base_tests(session)
    cleanup(session)


def run_l0_dla_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_models(session)
    run_base_tests(session)
    cleanup(session)


def run_l1_accuracy_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_datasets(session)
    train_model(session)
    run_accuracy_tests(session)
    cleanup(session)


def run_l1_int8_accuracy_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_datasets(session)
    train_model(session)
    finetune_model(session)
    run_int8_accuracy_tests(session)
    cleanup(session)


def run_l2_trt_compatibility_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_models(session)
    download_datasets(session)
    train_model(session)
    run_trt_compatibility_tests(session)
    cleanup(session)


def run_l2_multi_gpu_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_models(session)
    run_multi_gpu_tests(session)
    cleanup(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l0_api_tests(session):
    """When a developer needs to check correctness for a PR or something"""
    run_l0_api_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l0_dla_tests(session):
    """When a developer needs to check basic api functionality using host dependencies"""
    run_l0_dla_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l1_accuracy_tests(session):
    """Checking accuracy performance on various usecases"""
    run_l1_accuracy_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l1_int8_accuracy_tests(session):
    """Checking accuracy performance on various usecases"""
    run_l1_int8_accuracy_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l2_trt_compatibility_tests(session):
    """Makes sure that TensorRT Python and Torch-TensorRT can work together"""
    run_l2_trt_compatibility_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l2_multi_gpu_tests(session):
    """Makes sure that Torch-TensorRT can operate on multi-gpu systems"""
    run_l2_multi_gpu_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def download_test_models(session):
    """Grab all the models needed for testing"""
    try:
        import torch
    except ModuleNotFoundError:
        install_deps(session)
    download_models(session)
