# List all available recipes
default:
    @just --list

# ── Docs ──────────────────────────────────────────────────────────────────────

# Build HTML documentation (manages deps via uv)
docs:
    cd docsrc && uv run --group docs --prerelease allow make html

# Clean docs build artifacts (does not touch the published docs/ directory)
docs-clean:
    rm -rf docsrc/_build docsrc/_cpp_api docsrc/_py_api docsrc/_tmp docsrc/tutorials/_rendered_examples

# Serve the already-built docs locally
docs-serve port="3000":
    python3 -m http.server {{port}} --directory docs

# Build docs then serve them
docs-build-serve port="3000": docs
    python3 -m http.server {{port}} --directory docs
