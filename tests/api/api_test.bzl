
def api_test(name, visibility=None):
    native.cc_test(
	name = name,
	srcs = [name + ".cpp"],
	visibility = visibility,
	deps = [
	    "//tests/util",
	    "//core",
	    "@googletest//:gtest_main",
	],
	timeout="short"
    )

