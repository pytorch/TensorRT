#!/bin/sh
# Verify that the delegate imports the shared registry and can load from the wheel layout.
# Usage: check_imports_executorch_runtime.sh <readelf> <shared-object> [libexecutorch.so]
#            [expected-runpath] [manylinux-tag]

set -u
set -f
LC_ALL=C
export LC_ALL

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
    echo "FATAL: expected <readelf> <shared-object> [libexecutorch.so] [expected-runpath] [manylinux-tag]" >&2
    exit 1
fi
readelf_bin="$1"
target="$2"
runtime="${3-}"
expected_runpath="${4-}"
manylinux_tag="${5-}"

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

# An omitted option disables its check; an empty supplied value is a caller error.
for argument in "$@"; do
    [ -n "${argument}" ] || fail "supplied arguments must not be empty"
done
if [ "$#" -ge 5 ]; then
    case "${manylinux_tag}" in
        manylinux_2_28_x86_64|manylinux_2_39_aarch64) ;;
        *) fail "unsupported manylinux tag: ${manylinux_tag}" ;;
    esac
else
    echo "note: no manylinux tag given, so platform symbol-version checks are skipped" >&2
fi
if [ "$#" -lt 3 ]; then
    echo "note: no runtime given, so runtime export and symbol-version comparisons are skipped" >&2
fi

# auditwheel 6.8.2 manylinux-policy.json, limited to these two release architectures.
# Numeric gaps and named nodes are policy entries, not compiler-version ceilings.
policy_versions() {
    case "${manylinux_tag}" in
        manylinux_2_28_x86_64)
            cat <<'POLICY'
GLIBC 2.2.5 2.2.6 2.3 2.3.2 2.3.3 2.3.4 2.4 2.5 2.6 2.7 2.8 2.9 2.10 2.11 2.12 2.13 2.14 2.15 2.16 2.17 2.18 2.22 2.23 2.24 2.25 2.26 2.27 2.28
GLIBCXX 3.4 3.4.1 3.4.2 3.4.3 3.4.4 3.4.5 3.4.6 3.4.7 3.4.8 3.4.9 3.4.10 3.4.11 3.4.12 3.4.13 3.4.14 3.4.15 3.4.16 3.4.17 3.4.18 3.4.19 3.4.20 3.4.21 3.4.22 3.4.23 3.4.24
CXXABI 1.3 1.3.1 1.3.2 1.3.3 1.3.4 1.3.5 1.3.6 1.3.7 1.3.8 1.3.9 1.3.10 1.3.11 FLOAT128 TM_1
GCC 3.0 3.3 3.3.1 3.4 3.4.2 3.4.4 4.0.0 4.2.0 4.3.0 4.7.0 4.8.0 7.0.0
POLICY
            ;;
        manylinux_2_39_aarch64)
            cat <<'POLICY'
GLIBC 2.0 2.17 2.18 2.22 2.23 2.24 2.25 2.26 2.27 2.28 2.29 2.30 2.31 2.32 2.33 2.34 2.35 2.36 2.38 2.39 ABI_DT_RELR
GLIBCXX 3.4 3.4.1 3.4.2 3.4.3 3.4.4 3.4.5 3.4.6 3.4.7 3.4.8 3.4.9 3.4.10 3.4.11 3.4.12 3.4.13 3.4.14 3.4.15 3.4.16 3.4.17 3.4.18 3.4.19 3.4.20 3.4.21 3.4.22 3.4.23 3.4.24 3.4.25 3.4.26 3.4.27 3.4.28 3.4.29 3.4.30 3.4.31 3.4.32 3.4.33
CXXABI 1.3 1.3.1 1.3.2 1.3.3 1.3.4 1.3.5 1.3.6 1.3.7 1.3.8 1.3.9 1.3.10 1.3.11 1.3.12 1.3.13 1.3.14 1.3.15 TM_1
GCC 3.0 3.3 3.3.1 3.4 3.4.2 3.4.4 4.0.0 4.2.0 4.3.0 4.5.0 4.7.0 7.0.0 11.0 13.0.0 14.0 14.0.0
POLICY
            ;;
    esac | awk '{ for (i = 2; i <= NF; i++) print $1 "_" $i }'
}

versions() {
    printf '%s\n' "$1" |
        grep -oE '(GLIBCXX|CXXABI|GLIBC|GCC)_[A-Za-z0-9_.]+' | sort -u
}

needed_entries() {
    printf '%s\n' "$1" | sed -n 's/.*NEEDED.*\[\(.*\)\].*/\1/p'
}

# Check readelf's status before parsing; a pipeline would report the parser's status instead.
needed_of() {
    needed_dynamic=$("${readelf_bin}" -d "$1") || return 1
    needed_entries "${needed_dynamic}"
}

dyn=$("${readelf_bin}" -d "${target}") ||
    fail "could not inspect ${target} with ${readelf_bin}"
if ! printf '%s\n' "${dyn}" | grep -qE 'NEEDED.*\[libexecutorch\.so\]'; then
    fail "${target} has no DT_NEEDED on libexecutorch.so, so it would not bind to the shared backend registry"
fi
if ! printf '%s\n' "${dyn}" | grep -qE 'NEEDED.*\[libexecutorch_extension_cuda\.so\]'; then
    fail "${target} has no DT_NEEDED on libexecutorch_extension_cuda.so, so it may carry a private CUDA stream implementation"
fi
if ! printf '%s\n' "${dyn}" | grep -qE 'NEEDED.*\[libstdc\+\+\.so\.[0-9]+\]'; then
    fail "${target} has no DT_NEEDED on libstdc++, so it is not linked against the shared C++ runtime"
fi

runpath=$(printf '%s\n' "${dyn}" | sed -n '/RUNPATH/s/.*\[\(.*\)\].*/\1/p')
# GNU/LLVM print parenthesized tags; elfutils prints bare tags.
if printf '%s\n' "${dyn}" | grep -qE '[[:space:]]\(?RPATH\)?[[:space:]]'; then
    fail "${target} carries DT_RPATH rather than DT_RUNPATH"
fi
[ -n "${runpath}" ] || fail "${target} carries no RUNPATH"

cuda_needed=$(needed_entries "${dyn}" | grep -E '^libcudart\.so\.' | sort -u)
if [ -n "${cuda_needed}" ]; then
    case "${cuda_needed}" in
        libcudart.so.13) ;;
        *) fail "${target} needs ${cuda_needed}, but this delegate requires CUDA 13" ;;
    esac
    case ":${runpath}:" in
        *':$ORIGIN/../../nvidia/cu13/lib:'*) ;;
        *) fail "${target} needs ${cuda_needed}, but the RUNPATH carries no nvidia/cu13/lib" ;;
    esac
fi

if [ "$#" -ge 4 ]; then
    if [ "${runpath}" != "${expected_runpath}" ]; then
        fail "${target} carries a RUNPATH the build did not ask for:
  expected: ${expected_runpath}
  actual:   ${runpath}"
    fi
elif ! printf '%s\n' "${runpath}" | tr ':' '\n' | grep -Fxq '$ORIGIN/../../executorch/lib'; then
    fail "${target} has a RUNPATH but not \$ORIGIN/../../executorch/lib"
fi
absolute=$(printf '%s\n' "${runpath}" | tr ':' '\n' | grep -v '^\$ORIGIN\(/\|$\)' || true)
if [ -n "${absolute}" ]; then
    fail "${target} carries RUNPATH entries that are not relative to the artifact:
${absolute}"
fi

# Unversioned C++ helpers evade the version-node check and must come from libstdc++_nonshared.a.
dyn_syms=$("${readelf_bin}" --dyn-syms -W "${target}") ||
    fail "could not read the dynamic symbols of ${target} with ${readelf_bin}"
unversioned_cxx=$(printf '%s\n' "${dyn_syms}" |
    awk '($4 == "FUNC" || $4 == "OBJECT") && ($7 == "UND" || $7 == "UNDEF") && $8 !~ /@/ && $8 ~ /^(_ZNSt|_ZNKSt|_ZSt|_ZTVNSt|_ZTINSt|_ZN9__gnu_cxx|_ZTVN9__gnu_cxx|_ZTIN9__gnu_cxx)/ { print $8 }')
if [ -n "${unversioned_cxx}" ]; then
    fail "${target} has unversioned undefined C++ runtime symbols:
${unversioned_cxx}"
fi

syms=$("${readelf_bin}" -Ws "${target}") ||
    fail "could not read the symbols of ${target}"
register_backend='_ZN10executorch7runtime16register_backendERKNS0_7BackendE'
if ! printf '%s\n' "${syms}" | grep -q "${register_backend}"; then
    fail "${target} does not reference register_backend at all, so it registers no backend"
fi
if printf '%s\n' "${syms}" | grep "${register_backend}" | grep -qvE '[[:space:]]UND(EF)?[[:space:]]'; then
    fail "${target} defines register_backend instead of importing it, so it would register into a private registry"
fi

if [ "$#" -ge 3 ]; then
    [ -f "${runtime}" ] || fail "cannot compare symbol versions: ${runtime} does not exist"
    target_versions=$("${readelf_bin}" -V "${target}") ||
        fail "could not read symbol versions of ${target} with ${readelf_bin}"
    if ! versions "${target_versions}" | grep -q '^CXXABI_[0-9]'; then
        fail "${target} declares no CXXABI requirement, so it is under-linked or symbol versions could not be read"
    fi

    runtime_syms=$("${readelf_bin}" -Ws "${runtime}") ||
        fail "cannot read the symbol table of ${runtime}"
    if ! printf '%s\n' "${runtime_syms}" |
        grep -qE "(GLOBAL|WEAK)[[:space:]]+DEFAULT[[:space:]]+[0-9]+[[:space:]]+${register_backend}$"; then
        fail "${runtime} does not export ${register_backend}, which ${target} imports"
    fi

    runtime_dir=$(dirname "${runtime}")
    # executorch.runtime imports portable_lib, which loads _C and its kernel/backend dependencies.
    # Include that graph, but not unrelated sibling libraries that could widen the untagged check.
    set +f
    set -- "${runtime_dir}"/../extension/pybindings/_C.*.so
    set -f
    pybindings="$1"
    [ -f "${pybindings}" ] || fail "could not find the pybindings extension under ${runtime_dir}/../extension/pybindings/"
    target_needed=$(needed_entries "${dyn}")
    pybindings_needed=$(needed_of "${pybindings}") || fail "could not read dependencies of ${pybindings}"
    worklist="$(basename "${runtime}")
${target_needed}
${pybindings_needed}"
    closure=""
    runtime_versions=""
    while [ -n "${worklist}" ]; do
        name=$(printf '%s\n' "${worklist}" | head -1)
        worklist=$(printf '%s\n' "${worklist}" | tail -n +2)
        [ -n "${name}" ] || continue
        case " ${closure} " in
            *" ${name} "*) continue ;;
        esac
        closure="${closure} ${name}"
        sibling="${runtime_dir}/${name}"
        [ -f "${sibling}" ] || continue
        [ "${sibling}" = "${target}" ] && continue
        sibling_needed=$(needed_of "${sibling}") || fail "could not read dependencies of ${sibling}"
        sibling_versions=$("${readelf_bin}" -V "${sibling}") || fail "could not read symbol versions of ${sibling}"
        worklist="${worklist}
${sibling_needed}"
        runtime_versions="${runtime_versions}
${sibling_versions}"
    done

    if ! printf '%s\n' "${runtime_versions}" | grep -q '[^[:space:]]'; then
        fail "could not read symbol versions beside ${runtime} with ${readelf_bin}"
    fi

    if [ -n "${manylinux_tag}" ]; then
        allowed=$(policy_versions)
        for node in $(versions "${target_versions}"); do
            printf '%s\n' "${allowed}" | grep -Fxq "${node}" ||
                fail "${target} requires ${node}, which ${manylinux_tag} does not allow (auditwheel 6.8.2)"
        done
    else
        # Without a platform tag, retain the conservative named-node comparison only.
        for node in $(versions "${target_versions}" | grep -E '_[A-Z][A-Z0-9_]*$'); do
            versions "${runtime_versions}" | grep -Fxq "${node}" ||
                fail "${target} requires symbol versions absent from the runtime dependency closure: ${node}"
        done
    fi
fi

exit 0
