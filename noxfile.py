# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

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

# Set the USE_HOST_DEPS=1 to use host dependencies for tests
USE_HOST_DEPS = 0 if not "USE_HOST_DEPS" in os.environ else os.environ["USE_HOST_DEPS"]
if USE_HOST_DEPS:
    print("Using dependencies from host python")

SUPPORTED_PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]

nox.options.sessions = [
    "l0_api_tests-" + "{}.{}".format(sys.version_info.major, sys.version_info.minor)
]


def install_deps(session):
    print("Installing deps")
    session.install("-r", os.path.join(TOP_DIR, "py", "requirements.txt"))
    session.install("-r", os.path.join(TOP_DIR, "tests", "py", "requirements.txt"))


def install_torch_trt(session):
    print("Installing latest torch-tensorrt build")
    session.chdir(TOP_DIR)
    session.run(
        "python", "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."
    )


def cleanup(session):
    target = [
        "examples/int8/training/vgg16/*.jit.pt",
        "examples/int8/training/vgg16/vgg16_ckpts",
        "examples/int8/training/vgg16/cifar-10-*",
        "examples/int8/training/vgg16/data",
    ]

    target = " ".join(x for x in [os.path.join(TOP_DIR, i) for i in target])
    session.run_always("bash", "-c", str("rm -rf ") + target, external=True)


def run_base_tests(session):
    print("Running basic tests")
    session.chdir(os.path.join(TOP_DIR, "tests/py/core"))
    tests = [
        ".",
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


def run_l0_api_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
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


def run_dynamo_model_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    run_dynamo_model_compile_tests(session)
    run_dynamo_model_export_tests(session)
    cleanup(session)


def run_l1_dynamo_tests(session):
    if not USE_HOST_DEPS:
        install_deps(session)
        install_torch_trt(session)
    # SKIP hub.py download model failure on CI
    # run_dynamo_model_tests(session)
    run_dynamo_partitioning_tests(session)
    run_dynamo_runtime_tests(session)
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
def l1_model_tests(session):
    """When a user needs to test the functionality of standard models compilation and results"""
    run_dynamo_model_tests(session)


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, reuse_venv=True)
def l1_dynamo_tests(session):
    """When a user needs to test the functionality of standard models compilation and results"""
    run_l1_dynamo_tests(session)
