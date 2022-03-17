# Contribution Guidelines

### Developing Torch-TensorRT

Do try to fill an issue with your feature or bug before filling a PR (op support is generally an exception as long as you provide tests to prove functionality). There is also a backlog (https://github.com/NVIDIA/Torch-TensorRT/issues) of issues which are tagged with the area of focus, a coarse priority level and whether the issue may be accessible to new contributors. Let us know if you are interested in working on a issue. We are happy to provide guidance and mentorship for new contributors. Though note, there is no claiming of issues, we prefer getting working code quickly vs. addressing concerns about "wasted work".

#### Communication

The primary location for discussion is GitHub issues and Github discussions. This is the best place for questions about the project and discussion about specific issues.

We use the PyTorch Slack for communication about core development, integration with PyTorch and other communication that doesn't make sense in GitHub issues. If you need an invite, take a look at the [PyTorch README](https://github.com/pytorch/pytorch/blob/master/README.md) for instructions on requesting one.

### Coding Guidelines

- We generally follow the coding guidelines used in PyTorch

    - Use the built in linting tools to ensure that your code matches the style guidelines
      ```sh
      # C++ Linting (After installing clang-format [Version 9.0.0])
      # Print non-conforming sections of code
      bazel run //tools/linter:cpplint_diff -- //...
      # Modify code to conform with style guidelines
      bazel run //tools/linter:cpplint -- //...

      # Python Linting
      # Print non-conforming sections of code
      bazel run //tools/linter:pylint_diff -- //...
      # Modify code to conform with style guidelines
      bazel run //tools/linter:pylint -- //...
      ```

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved

- Try to avoid commiting commented out code

- Minimize warnings (and no errors) from the compiler

- Make sure all converter tests and the core module testsuite pass

- New features should have corresponding tests or if its a difficult feature to test in a testing framework, your methodology for testing.

- Comment subtleties and design decisions

- Document hacks, we can discuss it only if we can find it

### Commits and PRs

- Try to keep pull requests focused (multiple pull requests are okay). Typically PRs should focus on a single issue or a small collection of closely related issue.

- Typically we try to follow the guidelines set by https://www.conventionalcommits.org/en/v1.0.0/ for commit messages for clarity. Again not strictly enforced.

- #### Sign Your Work
    We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

    Any contribution which contains commits that are not Signed-Off will not be accepted.

    To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

        $ git commit -s -m "Add cool feature."

    This will append the following to your commit message:

        Signed-off-by: Your Name <your@email.com>

    By doing this you certify the below:

        Developer Certificate of Origin
        Version 1.1

        Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
        1 Letterman Drive
        Suite D4700
        San Francisco, CA, 94129

        Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


        Developer's Certificate of Origin 1.1

        By making a contribution to this project, I certify that:

        (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

        (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

        (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

        (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.


Thanks in advance for your patience as we review your contributions; we do appreciate them!
