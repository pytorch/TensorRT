# Release Process

Here is the process we use for creating new releases of TRTorch

## Criteria for Release

While TRTorch is in alpha, patch versions are bumped sequentially on breaking changes in the compiler.

In beta TRTorch will get a minor version bump on breaking changes or siginficant new functionality in the compiler, patch version will be incremented
based on significant bug fixes.

Once TRTorch hits version 1.0.0, major versions are bumped on breaking API changes, breaking changes or significant new functionality in the compiler
will result in a minor version bump and sigificant bug fixes will result in a patch version change.

## Steps to Packaging a Release

1. Freeze master
    - One release criteria has been hit, master should be frozen as we go through the release process
        - All open PRs should be merged or closed before freeze (unless changes are explicitly non applicable for the current release)
2. Release Testing
    - Required, Python API and Optional Tests should pass on both x86_64 and aarch64
    - All checked in applications (cpp and python) should compile and work
3. Version bump PR
    - There should be a PR which will be the PR that bumps the actual version of the library, this PR should contain the following
        - Bump version in `py/setup.py`
        - Make sure dependency versions are updated in `py/requirements.txt`, `tests/py/requirements.txt` and `py/setup.py`
        - Bump version in `cpp/api/include/macros.h`
        - Add new link to doc versions in `docsrc/conf.py`
        - Generate frozen docs for new version
            - Set `docsrc/conf.py` version to new version (temporarily, return back to master after)
            - `make html VERSION=<NEW_VERSION>`
            - Reset `docsrc/conf.py` version
            - `make html`
        - Generate changelog
            - `conventional-changelog -p angular -s CHANGELOG.md -t <last version tag> -a`
4. Once PR is merged tag commit and start creating release on GitHub
    - Paste in Milestone information and Changelog information into release notes
    - Generate libtrtorch.tar.gz for the following platforms:
        - x86_64 cxx11-abi
        - x86_64 pre-cxx11-abi
        - TODO: Add cxx11-abi build for aarch64 when a manylinux container for aarch64 exists
    - Generate Python packages for Python 3.5/3.6/3.7/3.8 for x86_64
        - TODO: Build a manylinux container for aarch64
        - `docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh` generates all wheels
            - To build container `docker build -t build_trtorch_wheel .`