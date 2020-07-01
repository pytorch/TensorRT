package(default_visibility = ["//visibility:public"])

config_setting(
   name = "aarch64",
   values = {
	"cpu" : "aarch64",
   },
)

cc_library(
    name = "cudart",
    srcs = select({
	":aarch64": [
            "targets/aarch64-linux-gnu/lib/libcudart.so",
        ],
	"//conditions:default": [
	    "targets/x86_64-linux/lib/libcudart.so",
	]
    }),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
)

cc_library(
    name = "nvToolsExt",
    srcs = select({
        ":aarch64": [
            "targets/aarch64-linux-gnu/lib/libnvToolsExt.so.1",
        ],
        "//conditions:default": [
            "targets/x86_64-linux/lib/libnvToolsExt.so.1",
        ]
    }),
)

cc_library(
    name = "cuda",
    srcs = select({
        ":aarch64": glob([
            "targets/aarch64-linux-gnu/lib/**/lib*.so",
        ]),
        "//conditions:default": glob([
            "targets/x86_64-linux/lib/**/lib*.so",
        ])
    }),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

