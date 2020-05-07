# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
VERSION       ?=
SPHINXOPTS    ?=
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
	mkdir -p /tmp/trtorch_docs
	mv $(DESTDIR)/v* /tmp/trtorch_docs
endif
	rm -r $(DESTDIR)/*
ifndef VERSION
	mv /tmp/trtorch_docs/v* $(DESTDIR)
	rm -rf /tmp/trtorch_docs
endif

html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	mkdir -p $(DESTDIR)
	cp -r $(BUILDDIR)/html/* $(DESTDIR)

.PHONY: help clean check_clean html Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
