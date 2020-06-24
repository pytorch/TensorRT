package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda",
    srcs = glob([
        "lib/**/*libcuda.so",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cudart",
    srcs = glob([
        "lib/**/*libcudart.so",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cublas",
    srcs = glob([
        "lib/**/*libcublas.so",
    ]),
    hdrs = glob([
        "include/**/*cublas*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)
