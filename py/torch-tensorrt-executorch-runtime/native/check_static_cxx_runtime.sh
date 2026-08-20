#!/bin/sh
# Verify a shipped shared object carries its own C++ runtime.
#
# This wheel has no auditwheel step behind it, so nothing else notices when an
# artifact picks up the build host's libstdc++. A failure here prints the NEEDED
# entries and the link command, because "has a dynamic libstdc++ dependency" on its
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

# A missing C++ runtime leaves mangled C++ symbols undefined. This used to look for
# exception_ptr::_M_addref alone, which only appears when something COPIES an
# exception_ptr: an object that never does passed the check and still failed to load
# with "undefined symbol: _ZTISt9exception". Measured on this toolchain, comparing a
# shared object linked with the static archive against the same one linked without it:
#
#   pattern                                 healthy   unloadable
#   UND .*_M_addref                            0          0      <- missed it
#   UND .*(_ZSt|_ZNSt|__cxa_|__gxx_personality) 6         46      <- false positive
#   UND .*(_ZS|_ZN|_ZT|__gxx_personality)      0         40      <- used here
#
# __cxa_atexit and __cxa_finalize come from glibc and are undefined in a healthy
# artifact, which is why the middle pattern cannot be used.
syms=$("${readelf_bin}" -Ws "${target}") ||
    fail "could not read symbols of ${target}"
if printf %s "${syms}" | grep -qE 'UND .*(_ZS|_ZN|_ZT|__gxx_personality)'; then
    fail "${target} has undefined C++ runtime symbols"
fi

exit 0
