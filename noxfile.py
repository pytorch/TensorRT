import os
import sys
from distutils.command.clean import clean

import nox

# Use system installed Python packages
PYT_PATH = (
    "/usr/local/lib/python3.12/dist-packages"
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

# Set the USE_PRE_CXX11=1 to use pre_cxx11_abi
USE_PRE_CXX11 = 0 if not "USE_PRE_CXX11" in os.environ else os.environ["USE_PRE_CXX11"]
if USE_PRE_CXX11:
    print("Using pre cxx11 abi")

# Set the USE_HOST_DEPS=1 to use host dependencies for tests
USE_HOST_DEPS = 0 if not "USE_HOST_DEPS" in os.environ else os.environ["USE_HOST_DEPS"]
if USE_HOST_DEPS:
    print("Using dependencies from host python")

# Set epochs to train VGG model for accuracy tests
EPOCHS = 25

SUPPORTED_PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]

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
    if USE_PRE_CXX11:
        session.run("python", "setup.py", "develop", "--use-pre-cxx11-abi")
    else:
        session.run("python", "setup.py", "develop")


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
            str(EPOCHS),
            env={"PYTHONPATH": PYT_PATH},
        )

        session.run_always(
            "python",
            "export_ckpt.py",
            "vgg16_ckpts/ckpt_epoch" + str(EPOCHS) + ".pth",
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
            str(EPOCHS),
        )

        session.run_always(
            "python", "export_ckpt.py", "vgg16_ckpts/ckpt_epoch" + str(EPOCHS) + ".pth"
        )


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
            str(EPOCHS),
            "--epochs",
            str(EPOCHS + 1),
            env={"PYTHONPATH": PYT_PATH},
        )

        # Export model
        session.run_always(
            "python",
            "export_qat.py",
            "vgg16_ckpts/ckpt_epoch" + str(EPOCHS + 1) + ".pth",
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
            str(EPOCHS),
            "--epochs",
            str(EPOCHS + 1),
        )

        # Export model
        session.run_always(
            "python",
            "export_qat.py",
            "vgg16_ckpts/ckpt_epoch" + str(EPOCHS + 1) + ".pth",
        )


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
    session.chdir(os.path.join(TOP_DIR, "tests/py/ts"))
    tests = [
        "api",
        "integrations/test_to_backend_api.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_dynamo_backend_tests(session):
    print("Running Dynamo core tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/dynamo/"))
    tests = [
        "backend",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_dynamo_converter_tests(session):
    print("Running Dynamo converter tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/dynamo/"))
    tests = [
        "conversion",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_dynamo_lower_tests(session):
    print("Running Dynamo lowering passes")
    session.chdir(os.path.join(TOP_DIR, "tests/py/dynamo/"))
    tests = ["lowering"]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_dynamo_partitioning_tests(session):
    print("Running Dynamo Partitioning tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/dynamo/"))
    tests = ["partitioning"]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_dynamo_runtime_tests(session):
    print("Running Dynamo Runtime tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/dynamo/"))
    tests = [
        "runtime",
    ]
    skip_tests = "-k not hw_compat"
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, skip_tests, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test, skip_tests)


def run_dynamo_model_compile_tests(session):
    print("Running model torch-compile tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/dynamo/models"))
    tests = [
        "test_models.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always(
                "python",
                test,
                "--ir",
                str("torch_compile"),
                env={"PYTHONPATH": PYT_PATH},
            )
        else:
            session.run_always("python", test, "--ir", str("torch_compile"))


def run_dynamo_model_export_tests(session):
    print("Running model torch-export tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/dynamo/models"))
    tests = ["test_models_export.py", "test_export_serde.py"]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always(
                "python", test, "--ir", str("dynamo"), env={"PYTHONPATH": PYT_PATH}
            )
        else:
            session.run_always("python", test, "--ir", str("dynamo"))


def run_accuracy_tests(session):
    print("Running accuracy tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/ts"))
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
    session.chdir(os.path.join(TOP_DIR, "tests/py/ts"))
    tests = [
        "ptq/test_ptq_to_backend.py",
        "ptq/test_ptq_dataloader_calibrator.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_trt_compatibility_tests(session):
    print("Running TensorRT compatibility tests")
    copy_model(session)
    session.chdir(os.path.join(TOP_DIR, "tests/py/ts"))
    tests = [
        "integrations/test_trt_intercompatibility.py",
        # "ptq/test_ptq_trt_calibrator.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_dla_tests(session):
    print("Running DLA tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/ts"))
    tests = [
        "hw/test_api_dla.py",
    ]
    for test in tests:
        if USE_HOST_DEPS:
            session.run_always("pytest", test, env={"PYTHONPATH": PYT_PATH})
        else:
            session.run_always("pytest", test)


def run_multi_gpu_tests(session):
    print("Running multi GPU tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/ts"))
    tests = [
        "hw/test_multi_gpu.py",
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


def run_l0_dynamo_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_backend_tests(session)
    run_dynamo_converter_tests(session)
    run_dynamo_lower_tests(session)
    cleanup(session)


def run_l0_dynamo_backend_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_backend_tests(session)
    cleanup(session)


def run_l0_dynamo_converter_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_converter_tests(session)
    cleanup(session)


def run_l0_dynamo_lower_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_lower_tests(session)
    cleanup(session)


def run_l0_dynamo_model_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_model_tests(session)
    cleanup(session)


def run_l0_dynamo_partitioning_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_partitioning_tests(session)
    cleanup(session)


def run_l0_dynamo_runtime_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_runtime_tests(session)
    cleanup(session)


def run_l0_dla_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_models(session)
    run_base_tests(session)
    cleanup(session)


def run_dynamo_model_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    download_models(session)
    run_dynamo_model_compile_tests(session)
    run_dynamo_model_export_tests(session)
    cleanup(session)


def run_l1_int8_accuracy_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    train_model(session)
    run_int8_accuracy_tests(session)
    cleanup(session)


def run_l1_dynamo_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_model_tests(session)
    run_dynamo_partitioning_tests(session)
    run_dynamo_runtime_tests(session)
    cleanup(session)


def run_l2_trt_compatibility_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
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
def l0_dynamo_tests(session):
    """When a developer needs to check correctness for a PR or something"""
    run_l0_dynamo_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l0_dynamo_backend_tests(session):
    """When a developer needs to check correctness for a PR or something"""
    run_l0_dynamo_backend_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l0_dynamo_converter_tests(session):
    """When a developer needs to check correctness for a PR or something"""
    run_l0_dynamo_converter_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l0_dynamo_lower_tests(session):
    """When a developer needs to check correctness for a PR or something"""
    run_l0_dynamo_lower_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l0_dla_tests(session):
    """When a developer needs to check basic api functionality using host dependencies"""
    run_l0_dla_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l1_model_tests(session):
    """When a user needs to test the functionality of standard models compilation and results"""
    run_dynamo_model_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l1_dynamo_tests(session):
    """When a user needs to test the functionality of standard models compilation and results"""
    run_l1_dynamo_tests(session)


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
