import os
import sys
import glob
import subprocess
import utils


def lint(target_files, color=True):
    failure = False
    cmd = ["black", "--diff"]
    if color:
        cmd += ["--color"]
    else:
        cmd += ["--no-color"]
    cmd += target_files
    output = subprocess.run(cmd)

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
        files = utils.glob_files(path, utils.VALID_PY_FILE_TYPES)
        if files != []:
            if lint(files, color):
                failure = True
    if failure:
        if color:
            print("\033[91mERROR:\033[0m Some files do not conform to style guidelines")
        else:
            print("ERROR: Some files do not conform to style guidelines")
        sys.exit(1)
