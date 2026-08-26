#!/bin/sh
# Verify a shipped shared object shares the process C++ runtime instead of carrying
# one of its own.
#
# These artifacts are loaded into a process that has already loaded libtorch and
# TensorRT, which both bring libstdc++.so.6. A second copy here does not isolate
# anything: libaoti_cuda_shims.so is the first DT_NEEDED of _portable_lib.so, so it
# leads the dlopen group in symbol search order and every object loaded with it
# resolves the C++ runtime against this wheel rather than against libstdc++.so.6.
# Two libstdc++ builds then share one process, and a locale facet built by one gets
# indexed with the other's std::locale::id, which lands a virtual call on the wrong
# slot.
#
# This wheel has no auditwheel step behind it, so nothing else notices. A failure
# prints the NEEDED entries and the link command, because the artifact property on
# its own says nothing about which input produced it.
#
# Usage: check_shared_cxx_runtime.sh <readelf> <shared-object> [link-command-file]

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
syms=$("${readelf_bin}" -WsD "${target}") ||
    fail "could not read dynamic symbols of ${target}"

# Defined, not undefined, and only symbols that libstdc++ alone implements. A plain
# _ZNSt or _ZSt prefix would reject a good artifact: every C++ shared object exports
# weak instantiations of std:: templates from its own translation units, and those are
# identical wherever they come from. The three families below are not. They are
# emitted only by libstdc++'s own translation units, so a definition here means a
# whole second runtime came in through libstdc++.a.
defined=$(printf %s\\n "${syms}" |
    awk '($5 == "GLOBAL" || $5 == "WEAK") && $7 != "UND" { print $8 }' |
    sed 's/@.*//' |
    grep -E '^(__cxa_(throw|rethrow|begin_catch|end_catch|allocate_exception|free_exception)$|_ZTVN10__cxxabiv1|_ZNS[tK]?6locale)')
if [ -n "${defined}" ]; then
    echo "--- libstdc++ symbols defined by ${target} ---" >&2
    printf %s\\n "${defined}" | head -20 >&2
    fail "${target} defines $(printf %s\\n "${defined}" | wc -l) libstdc++ symbols of its own"
fi

# The other half, and the original failure this guard was written for. Undefined
# runtime symbols are correct and expected once the runtime is shared, but only if
# something declares where they come from. Without the NEEDED entry the extension
# fails to import on a missing exception_ptr::_M_addref.
if printf %s\\n "${syms}" | grep -qE 'UND +(_ZNS[tK]|_ZS[tT]|__cxa_|_ZNKS[tK])' &&
        ! printf %s "${dyn}" | grep -qE 'NEEDED.*libstdc\+\+'; then
    fail "${target} references the C++ runtime but has no libstdc++ NEEDED entry"
fi

exit 0
