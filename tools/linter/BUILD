load("@devtools_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_binary(
    name = "cpplint",
    srcs = ["cpplint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        requirement("clang-format"),
    ],
)

py_binary(
    name = "cpplint_diff",
    srcs = ["cpplint_diff.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        requirement("clang-format"),
    ],
)

py_binary(
    name = "pylint",
    srcs = ["pylint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        requirement("black"),
    ],
)

py_binary(
    name = "pylint_diff",
    srcs = ["pylint_diff.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        requirement("black"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)
