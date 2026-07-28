import os
import sys
import glob
import subprocess
import utils
import clang_format


def lint(target_files, color=True):
    failure = False
    for f in target_files:
        with open("/tmp/changes.txt", "w") as changes:
            subprocess.run(
                [clang_format._get_executable("clang-format"), f], stdout=changes
            )
        args = ["git", "diff", "-u", "--exit-code"]
        if color:
            args += ["--color"]
        args += [f, "/tmp/changes.txt"]
        output = subprocess.run(args)
        if output.returncode != 0:
            failure = True
    return failure


if __name__ == "__main__":
    BAZEL_ROOT = utils.find_bazel_root()
    color = True
    if "--no-color" in sys.argv:
        sys.argv.remove("--no-color")
        color = False

    projects = utils.CHECK_PROJECTS(sys.argv[1:])
    if "//..." in projects:
        projects = [
            p.replace(BAZEL_ROOT, "/")[:-1] for p in glob.glob(BAZEL_ROOT + "/*/")
        ]
        projects = [p for p in projects if p not in utils.BLACKLISTED_BAZEL_TARGETS]

    failure = False
    for p in projects:
        if p.endswith("/..."):
            p = p[:-4]
        path = BAZEL_ROOT + "/" + p[2:]
        files = utils.glob_files(path, utils.VALID_CPP_FILE_TYPES)
        if files != []:
            if lint(files, color):
                failure = True
    if failure:
        if color:
            print("\033[91mERROR:\033[0m Some files do not conform to style guidelines")
        else:
            print("ERROR: Some files do not conform to style guidelines")
        sys.exit(1)
