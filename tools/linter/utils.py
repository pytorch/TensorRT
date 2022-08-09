import os
import sys
import glob
import subprocess

BLACKLISTED_BAZEL_TARGETS = [
    "//experiments",
    "//tools",
    "//docker",
    "//third_party",
    "//bazel-bin",
    "//bazel-genfiles",
    "//bazel-out",
    "//bazel-TRTorch",
    "//bazel-Torch-TensorRT",
    "//bazel-torch-tensorrt",
    "//bazel-workspace",
    "//bazel-tensorrt",
    "//bazel-TensorRT",
    "//bazel-testlogs",
    "//py/build",
    "//bazel-project",
    "//py/dist",
    "//py/trtorch.egg-info",
    "//py/wheelhouse",
    "//examples",
    "//docsrc",
    "//docs",
]

VALID_CPP_FILE_TYPES = [".cpp", ".cc", ".c", ".cu", ".hpp", ".h", ".cuh"]
VALID_PY_FILE_TYPES = [".py"]


def CHECK_PROJECTS(projs):
    for p in projs:
        if p[:2] != "//":
            sys.exit(p + " is not a valid bazel target")
    return projs


def find_bazel_root():
    """
    Finds the root directory of the bazel space
    """
    curdir = os.path.dirname(os.path.realpath(__file__))
    while 1:
        if os.path.exists(curdir + "/WORKSPACE"):
            return curdir
        if curdir == "/":
            sys.exit("Error: was unable to find a bazel workspace")
        curdir = os.path.dirname(curdir)


def glob_files(project, file_types):
    files = []
    for t in file_types:
        files += glob.glob(project + "/**/*" + t, recursive=True)
    return files
