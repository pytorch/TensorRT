#!/bin/sh
# Verify a shipped shared object carries its own C++ runtime.
#
# This wheel has no auditwheel step behind it, so nothing else notices when an
# artifact picks up the build host's libstdc++. A failure here prints the dynamic
# section and the link command, because "has a dynamic libstdc++ dependency" on its
# own says nothing about which input put it there.
#
# Usage: check_static_cxx_runtime.sh <readelf> <shared-object> [link-command-file]

set -u

readelf_bin="$1"
target="$2"
link_txt="${3:-}"

fail() {
    echo "FATAL: $*" >&2
    echo "--- NEEDED entries of ${target} ---" >&2
    "${readelf_bin}" -d "${target}" 2>&1 | grep NEEDED >&2 ||
        echo "(none, or readelf could not read it)" >&2
    if [ -n "${link_txt}" ] && [ -f "${link_txt}" ]; then
        echo "--- link command ---" >&2
        cat "${link_txt}" >&2
    else
        echo "--- link command unavailable (${link_txt:-no path given}) ---" >&2
    fi
    exit 1
}

dyn=$("${readelf_bin}" -d "${target}") ||
    fail "could not inspect ${target} with ${readelf_bin}"
if printf %s "${dyn}" | grep -qE 'NEEDED.*libstdc\+\+'; then
    fail "${target} has a dynamic libstdc++ dependency"
fi

# -static-libstdc++ alone has silently left this undefined before, when the target was
# linked by the C driver instead of the C++ one.
syms=$("${readelf_bin}" -Ws "${target}") ||
    fail "could not read symbols of ${target}"
if printf %s "${syms}" | grep -qE 'UND .*_M_addref'; then
    fail "${target} has an undefined exception_ptr::_M_addref"
fi

exit 0
