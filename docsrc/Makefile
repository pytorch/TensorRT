# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
VERSION       ?=
SPHINXOPTS    ?=
SPHINXAPIDOC  ?= sphinx-apidoc
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DESTDIR       = ../docs/$(VERSION)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

check_clean:
	@echo -n "Are you sure? Will remove all docs files in $(DESTDIR) [y/N] " && read ans && [ $${ans:-N} = y ]

clean: check_clean
	rm -rf $(BUILDDIR)/*
ifndef VERSION
	rm -rf /tmp/torchtrt_docs
	mkdir -p /tmp/torchtrt_docs
	mv $(DESTDIR)/v* /tmp/torchtrt_docs
endif
	rm -r $(DESTDIR)/*
ifndef VERSION
	mv /tmp/torchtrt_docs/v* $(DESTDIR)
	rm -rf /tmp/torchtrt_docs
endif
	rm -rf $(SOURCEDIR)/_cpp_api
	rm -rf $(SOURCEDIR)/_notebooks
	rm -rf $(SOURCEDIR)/_py_api
	rm -rf $(SOURCEDIR)/_build
	rm -rf $(SOURCEDIR)/_tmp

html:
#	mkdir -p $(SOURCEDIR)/_notebooks
#	cp -r $(SOURCEDIR)/../notebooks/*.ipynb $(SOURCEDIR)/_notebooks
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	mkdir -p $(DESTDIR)
	cp -r $(BUILDDIR)/html/* $(DESTDIR)
	touch $(DESTDIR)/.nojekyll

.PHONY: help clean check_clean html Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
