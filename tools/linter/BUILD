load("@pylinter_deps//:requirements.bzl", "requirement")

py_binary(
    name = "cpplint",
    srcs = ["cpplint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

py_binary(
    name = "cpplint_diff",
    srcs = ["cpplint_diff.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

py_binary(
    name = "pylint",
    srcs = ["pylint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        requirement("yapf"),
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
        requirement("yapf"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)
