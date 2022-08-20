import os
import sys
import glob
import subprocess
import utils
import pwd


def lint(user, target_files, change_file=True):
    out = subprocess.run(["black", "--check", "-q"] + target_files)

    if change_file and out.returncode != 0:
        cmd = ["black", "-q"]
        print(
            "\033[93mWARNING:\033[0m This command is modifying your files with the recommended linting, you should review the changes before committing"
        )

        cmd += target_files

        for f in target_files:
            subprocess.run(cmd)
            subprocess.run(["chown", user + ":" + user, f])
            subprocess.run(["chmod", "644", f])


if __name__ == "__main__":
    BAZEL_ROOT = utils.find_bazel_root()
    USER = pwd.getpwuid(os.getuid())[0]
    projects = utils.CHECK_PROJECTS(sys.argv[1:])
    if "//..." in projects:
        projects = [
            p.replace(BAZEL_ROOT, "/")[:-1] for p in glob.glob(BAZEL_ROOT + "/*/")
        ]
        projects = [p for p in projects if p not in utils.BLACKLISTED_BAZEL_TARGETS]

    for p in projects:
        if p.endswith("/..."):
            p = p[:-4]
        path = BAZEL_ROOT + "/" + p[2:]
        files = utils.glob_files(path, utils.VALID_PY_FILE_TYPES)
        if files != []:
            lint(USER, files)
