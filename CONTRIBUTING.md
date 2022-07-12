# Contribution Guidelines

### Developing Torch-TensorRT

Do try to fill an issue with your feature or bug before filling a PR (op support is generally an exception as long as you provide tests to prove functionality). There is also a backlog (https://github.com/pytorch/TensorRT/issues) of issues which are tagged with the area of focus, a coarse priority level and whether the issue may be accessible to new contributors. Let us know if you are interested in working on a issue. We are happy to provide guidance and mentorship for new contributors. Though note, there is no claiming of issues, we prefer getting working code quickly vs. addressing concerns about "wasted work".

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

- We require that all contributors sign CLA for submitting PRs. In order for us to review and merge your suggested changes, please sign at https://code.facebook.com/cla. If you are contributing on behalf of someone else (eg your employer), the individual CLA may not be sufficient and your employer may need to sign the corporate CLA.


Thanks in advance for your patience as we review your contributions; we do appreciate them!
