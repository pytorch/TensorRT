# Building the Documentation

We use Sphinx, Doxygen and pandoc for documentation, so begin by installing the dependencies:

```
apt install doxygen pandoc
```

```
pip3 install --user -r requirements.txt
```

Then you just need to run the Makefile to generate the docs in `../docs`

There are two main relevant commands (other sphinx make commands are available)

### `make html`

> Make sure any time there is an API change you regenerate the master docs

`make html` builds the docs for the current branch and replaces the existing master branch documentation.

You can specify a version to build docs when cutting a release like this `make html VERSION=v0.0.1`
This will place the documentation in `../docs/v0.0.1`, then you can update the `conf.py` to include
the new version documentation in the dropdown.

### `make clean`

`make clean` will only delete the master documentation, not any of the version documentation

To clean version documentation use `make clean VERSION=v0.0.1`
