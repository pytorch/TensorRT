load(":defs.bzl", "dep_collection")

package(default_visibility = ["//visibility:public"])

dep_collection(
    name = "compute_libs",
    build_setting_default = "default",
)

config_setting(
    name = "driveos",
    flag_values = {
        ":compute_libs": "driveos",
    },
)
