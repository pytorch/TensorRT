#!/bin/sh
# Verify the TensorRT delegate binds to the shipped ExecuTorch runtime rather than to a
# private copy of it.
#
# Registration happens in a static initializer that calls register_backend, and there is
# exactly one registry that matters: the one inside the libexecutorch.so the user's ExecuTorch
# already loaded. So the delegate has to IMPORT that symbol, not define it. A build that
# statically absorbed the runtime would link cleanly, load cleanly, register into its own
# private table, and then report the backend as unavailable with nothing to point at.
#
# Usage: check_imports_executorch_runtime.sh <readelf> <shared-object> [libexecutorch.so]
#            [expected-runpath] [manylinux-tag]

set -u

readelf_bin="$1"
target="$2"
runtime="${3:-}"
expected_runpath="${4:-}"
# The manylinux tag the wheel ships under, for example manylinux_2_28 or manylinux_2_39. Empty skips
# the symbol version ceiling, since there is then nothing to compare against.
glibc_floor="${5:-}"

# The highest version of each family a manylinux platform guarantees a host provides. Taken from the
# toolchain the matching quay.io/pypa image carries, not derived from the glibc number.
floor_for() {
    case "$2" in
        *_2_28) case "$1" in
                    GLIBC) echo "2.28" ;; GLIBCXX) echo "3.4.25" ;;
                    CXXABI) echo "1.3.11" ;; GCC) echo "7.0.0" ;;
                esac ;;
        *_2_34) case "$1" in
                    GLIBC) echo "2.34" ;; GLIBCXX) echo "3.4.29" ;;
                    CXXABI) echo "1.3.13" ;; GCC) echo "7.0.0" ;;
                esac ;;
        *_2_39) case "$1" in
                    GLIBC) echo "2.39" ;; GLIBCXX) echo "3.4.32" ;;
                    CXXABI) echo "1.3.15" ;; GCC) echo "7.0.0" ;;
                esac ;;
    esac
}
if [ -z "${runtime}" ]; then
    # Everything else in this script fails closed, so say when half of it is not running rather
    # than exiting 0 as though the artifact had been fully checked.
    echo "note: no runtime given, so the symbol-version comparison is skipped" >&2
fi

fail() {
    echo "FATAL: $*" >&2
    echo "--- NEEDED entries of ${target} ---" >&2
    "${readelf_bin}" -d "${target}" 2>&1 | grep NEEDED >&2 ||
        echo "(none, or readelf could not read it)" >&2
    exit 1
}

dyn=$("${readelf_bin}" -d "${target}") ||
    fail "could not inspect ${target} with ${readelf_bin}"
if ! printf %s "${dyn}" | grep -qE 'NEEDED.*\[libexecutorch\.so\]'; then
    fail "${target} has no DT_NEEDED on libexecutorch.so, so the loader would not bind it to the runtime that owns the backend registry"
fi
# The CUDA extension is linked PRIVATE in native/CMakeLists.txt, so a shared executorch::extension_cuda
# leaves a DT_NEEDED here. Its absence means the imported target resolved to a static archive and the
# delegate absorbed a private copy of the caller-stream implementation instead of sharing ExecuTorch's;
# registration and the import checks above still pass, and the mixed-delegate case then runs on a
# different stream. __init__.py's CPU-wheel diagnosis also assumes this DT_NEEDED exists. Require it so
# a static link is caught in the build rather than at a user's execute().
if ! printf %s "${dyn}" | grep -qE 'NEEDED.*\[libexecutorch_extension_cuda\.so\]'; then
    fail "${target} has no DT_NEEDED on libexecutorch_extension_cuda.so, so executorch::extension_cuda linked as a static archive and the delegate carries a private CUDA stream implementation instead of sharing ExecuTorch's"
fi

# libstdc++ must be present as a normal dynamic dependency, the same shape as ExecuTorch's own
# delegates. The build toolchain is newer than the libstdc++.so.6 on a user's machine, and an
# optimized build emits out-of-line calls into the newer runtime (for example
# std::string::_M_replace_cold). Those newer helpers are pulled statically from the toolchain's
# libstdc++_nonshared.a (see native/CMakeLists.txt), while the old, stable symbols resolve against
# the system libstdc++.so.6 named in DT_NEEDED. Its absence would mean the C++ runtime was linked
# some other way than the intended one, so require it.
if ! printf %s "${dyn}" | grep -qE 'NEEDED.*\[libstdc\+\+\.so\.[0-9]+\]'; then
    fail "${target} has no DT_NEEDED on libstdc++, so it was not linked against the system C++ runtime the way ExecuTorch's own delegates are"
fi
# The failure mode this stack hit: an UNVERSIONED undefined libstdc++ symbol. A newer-toolchain helper
# such as std::string::_M_replace_cold, left unresolved, appears as an UND with no @GLIBCXX version and
# so is invisible to the symbol-version ceiling comparison below. libstdc++_nonshared.a must have
# supplied it. Every remaining undefined libstdc++/libgcc symbol must carry a version, or the delegate
# can fail to load on a host whose libstdc++ lacks the unversioned name. OBJECT is checked alongside
# FUNC so an unversioned vtable or typeinfo is caught too, and the names are anchored to the std and
# __gnu_cxx mangling prefixes so a legitimately unversioned symbol from another library is not flagged.
#
# readelf is read into a variable first, with its own status checked, rather than piped straight into
# awk. A pipeline reports awk's status, not readelf's, so a readelf that errors would leave an empty
# result that reads as "clean" and fails open. This check is the only one written for the exact symbol
# this stack shipped broken, so it must fail closed when it cannot see the symbols.
dyn_syms=$("${readelf_bin}" --dyn-syms -W "${target}") ||
    fail "could not read the dynamic symbols of ${target} with ${readelf_bin}"
unversioned_cxx=$(printf '%s\n' "${dyn_syms}" |
    awk '($4 == "FUNC" || $4 == "OBJECT") && ($7 == "UND" || $7 == "UNDEF") && $8 !~ /@/ && $8 ~ /^(_ZNSt|_ZNKSt|_ZSt|_ZTVNSt|_ZTINSt|_ZN9__gnu_cxx|_ZTVN9__gnu_cxx|_ZTIN9__gnu_cxx)/ { print $8 }')
if [ -n "${unversioned_cxx}" ]; then
    printf 'FATAL: %s has unversioned undefined C++ runtime symbols (newer-toolchain helpers not supplied by libstdc++_nonshared.a):\n%s\n' "${target}" "${unversioned_cxx}" >&2
    exit 1
fi

# The CUDA half of the same contract. The RUNPATH ships one directory per CUDA layout, because the
# two majors package their runtime differently: cu13 wheels install nvidia/cu13/lib carrying
# libcudart.so.13, and cu12 wheels install nvidia/cuda_runtime/lib carrying libcudart.so.12. Either
# is fine, since torch-tensorrt publishes both channels. What is NOT fine is a major with no matching
# RUNPATH entry, which links cleanly and then finds nothing at load time.
# require_supported_cuda() in setup.py reads torch.version.cuda, which is not necessarily the toolkit
# Bazel handed CMake, so check the artifact itself.
cuda_needed=$(printf '%s\n' "${dyn}" | grep -oE 'libcudart\.so\.[0-9]+' | sort -u || true)
if [ -n "${cuda_needed}" ]; then
    case "${cuda_needed}" in
        libcudart.so.12) cuda_dir="nvidia/cuda_runtime/lib" ;;
        libcudart.so.13) cuda_dir="nvidia/cu13/lib" ;;
        *)
            echo "FATAL: ${target} needs ${cuda_needed}, which this wheel ships no runtime directory for." >&2
            exit 1
            ;;
    esac
    case "${expected_runpath}" in
        *"${cuda_dir}"*) ;;
        *)
            echo "FATAL: ${target} needs ${cuda_needed}, but the RUNPATH carries no ${cuda_dir}, so the loader would not find it." >&2
            exit 1
            ;;
    esac
fi

syms=$("${readelf_bin}" -Ws "${target}") ||
    fail "could not read the symbols of ${target}"
# Mangled, because that is what is in .dynsym: executorch::runtime::register_backend.
register_backend='_ZN10executorch7runtime16register_backendERKNS0_7BackendE'
if ! printf %s "${syms}" | grep -q "${register_backend}"; then
    fail "${target} does not reference register_backend at all, so it registers no backend"
fi
# UND from binutils, UNDEF from elfutils, since the error above invites any readelf.
if printf %s "${syms}" | grep "${register_backend}" | grep -qvE '\bUND(EF)?\b'; then
    fail "${target} defines register_backend instead of importing it, so it would register into a private registry that nothing queries"
fi

# The delegate must not need a newer runtime than the ExecuTorch it loads beside. Its libstdc++
# helpers come from the toolchain's nonshared archive (checked above), so its GLIBCXX and CXXABI
# needs stay at the old floor, but GLIBC and the symbol versions reached through the ExecuTorch
# dependency graph still resolve against whatever the host has, and a delegate built with a newer
# toolchain can require a version the host lacks while ExecuTorch itself loads fine. That failure
# appears at load time on the user's machine and not in this build, because the build container's
# own toolchain libraries are on LD_LIBRARY_PATH.
#
# Comparing against libexecutorch.so rather than a hardcoded floor keeps this honest when the
# pin moves: the requirement is only ever "no worse than what ExecuTorch already asks for".
if [ -n "${runtime}" ]; then
    [ -f "${runtime}" ] ||
        fail "cannot compare symbol versions: ${runtime} does not exist"

    # Read once, up front, so readelf's exit status is checked in the parent shell. A `|| fail`
    # inside a function only ever reached through $(...) exits the command substitution subshell
    # and leaves the parent running with an empty string, which is the opposite of failing closed.
    target_versions=$("${readelf_bin}" -V "${target}") ||
        fail "could not read symbol versions of ${target} with ${readelf_bin}"
    # The ceiling is what the host must already provide for the pinned ExecuTorch and the delegate
    # to load, so it is libexecutorch.so, the components ExecuTorch pulls in when it is imported, and
    # the libraries the delegate links directly, not every file that happens to share the directory.
    # A wheel is built by more than one toolchain: in the pinned one, libexecutorch.so tops out at
    # GLIBCXX_3.4.21 and GCC_3.0 while its optimized kernels need GLIBCXX_3.4.22 and GCC_4.0.0 and its
    # xnnpack backend needs GCC_3.4. Those two are reached only through the pybindings extension, not
    # from libexecutorch.so, so the closure below is seeded from the pybindings extension as well as
    # the delegate and libexecutorch.so. It keeps in scope what the process actually loads, while an
    # unrelated sibling that nothing in the closure needs stays out of it. libexecutorch_extension_cuda.so
    # is kept for the same reason: the delegate needs it and libexecutorch.so does not, so it enters
    # from the delegate's own DT_NEEDED. A directory glob let an unrelated sibling raise the ceiling
    # and admit a delegate the pinned ExecuTorch cannot load.
    #
    # The version-reading loop below and the closure walk both skip the target itself. This is
    # defensive rather than load-bearing: nothing in the tree DT_NEEDEDs the delegate, so its
    # basename never enters the closure and the skip does not fire on any artifact shipped today.
    # It is kept so that a future library that does link the delegate cannot fold the delegate's
    # own requirements into the ceiling and make the comparison self-satisfying.
    runtime_dir=$(dirname "${runtime}")
    needed_of() {
        "${readelf_bin}" -d "$1" 2>/dev/null |
            sed -n 's/.*NEEDED.*\[\(.*\)\].*/\1/p'
    }
    # Breadth-first over DT_NEEDED, rooted at the delegate, libexecutorch.so, and the pybindings
    # extension, resolved against the files actually shipped beside the runtime. POSIX sh has no
    # sets, so track visited names in a space-delimited string.
    #
    # The pybindings extension is seeded because executorch/__init__.py loads it (a normal Python
    # import) before the delegate is dlopened, so whatever C++ runtime version it pulls in is already
    # present in the process by the time the delegate loads. It is not beside libexecutorch.so; it
    # sits in executorch/extension/pybindings/. Its DT_NEEDED graph reaches the optimized-kernels and
    # xnnpack libraries, which do live beside libexecutorch.so and top out higher than libexecutorch.so
    # itself (GLIBCXX_3.4.22 vs 3.4.21). Leaving it out computed a ceiling below what the process
    # already provides and rejected a delegate that would in fact load.
    closure=""
    pybindings=$(ls "${runtime_dir}"/../extension/pybindings/_C.*.so 2>/dev/null | head -1)
    if [ -z "${pybindings}" ]; then
        fail "could not find the pybindings extension _C.*.so under ${runtime_dir}/../extension/pybindings/; the symbol-version ceiling below is seeded from it, so a missing extension would silently narrow this guard"
    fi
    worklist="$(basename "${runtime}")
$(needed_of "${target}")$([ -n "${pybindings}" ] && printf '\n%s' "$(needed_of "${pybindings}")")"
    while [ -n "${worklist}" ]; do
        name=$(printf '%s\n' "${worklist}" | head -1)
        worklist=$(printf '%s\n' "${worklist}" | tail -n +2)
        [ -z "${name}" ] && continue
        case " ${closure} " in
        *" ${name} "*) continue ;;
        esac
        closure="${closure} ${name}"
        sibling="${runtime_dir}/${name}"
        if [ -f "${sibling}" ] && [ "${sibling}" != "${target}" ]; then
            worklist="${worklist}
$(needed_of "${sibling}")"
        fi
    done
    runtime_versions=$(
        for name in ${closure}; do
            sibling="${runtime_dir}/${name}"
            [ -f "${sibling}" ] || continue
            [ "${sibling}" = "${target}" ] && continue
            "${readelf_bin}" -V "${sibling}"
        done
    )
    [ -n "${runtime_versions}" ] ||
        fail "could not read symbol versions beside ${runtime} with ${readelf_bin}"

    # GLIBC too, not just the C++ families: a delegate built against a newer glibc than the
    # runtime it pairs with fails on the same hosts, and glibc is the most common of the three.
    # Two node shapes, because the named ones carry no dotted version: CXXABI_TM_1 and
    # CXXABI_FLOAT128 are real requirements, and a pattern demanding digits after the underscore
    # drops them silently.
    versions() {
        printf %s "$1" |
            grep -oE '(GLIBCXX|CXXABI|GLIBC|GCC)_([0-9]+(\.[0-9]+)*|[A-Z][A-Z0-9_]*)' | sort -u
    }
    # Highest version of one family, ordered numerically field by field rather than as text, so
    # 3.4.9 does not outrank 3.4.21. Named nodes sort first, not last: a non-numeric first field
    # compares as 0, so CXXABI_TM_1 lands below CXXABI_1.3. That is why they cannot be checked
    # here at all and are compared as a set above.
    highest() {
        versions "$1" | grep "^$2_" | sed "s/^$2_//" |
            sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | tail -1
    }

    # The delegate links libstdc++ dynamically, so it declares a CXXABI requirement. Its absence
    # would mean the C++ runtime was not linked the intended way (for example the toolchain's
    # as-needed -lstdc++ was dropped and nothing replaced it), which would leave the floor comparison
    # with nothing to check. Require the family the delegate cannot legitimately be missing.
    if [ -z "$(highest "${target_versions}" CXXABI)" ]; then
        fail "${target} declares no CXXABI requirement, so it is under-linked against the C++ runtime or symbol versions could not be read, and the floor comparison cannot be trusted"
    fi

    # Named nodes only, checked as a set. Symbol versioning is backward compatible for numbered
    # nodes (a runtime declaring GLIBC_2.27 satisfies a delegate needing GLIBC_2.4), so exact
    # set membership is the wrong test for those and the per-family maximum below is the right
    # one. It is the only available test for CXXABI_TM_1 and CXXABI_FLOAT128, which carry no
    # version number, sort below every numbered node, and so are invisible to a maximum. Getting
    # this wrong rejected the delegate against the very runtime it is pinned to: two libraries
    # from one ExecuTorch wheel that load together every day fail an exact-set check.
    named() {
        versions "$1" | grep -E '_[A-Z][A-Z0-9_]*$'
    }
    missing=$(
        for node in $(named "${target_versions}"); do
            named "${runtime_versions}" | grep -Fxq "${node}" || echo "${node}"
        done
    )
    if [ -n "${missing}" ]; then
        echo "FATAL: ${target} requires symbol versions that $(basename "${runtime}") does not:" >&2
        printf '%s\n' "${missing}" | sed 's/^/  /' >&2
        echo "The delegate would fail to load on hosts where the ExecuTorch it pairs with loads fine." >&2
        exit 1
    fi

    # And that the runtime actually defines this one symbol, not just that the delegate imports it.
    # Checking only the import side accepted a runtime with no such export: the build stayed green
    # and the failure moved to an undefined symbol at load time, which is the "backend unavailable
    # for no visible reason" this script exists to prevent.
    #
    # One symbol, not every symbol the delegate imports. A pin bump that drops some other export
    # reaches the same load-time failure and is not caught here, because at this point in the
    # build there is no sibling site-packages to resolve the rest against. The wheel-build step
    # runs "ldd -r" against the installed layout, which is where that gap is closed.
    runtime_syms=$("${readelf_bin}" -Ws "${runtime}") ||
        fail "cannot read the symbol table of ${runtime}"
    # A defined export carries a section index where an import carries UND, so requiring a digit
    # there is what separates the two. -Ws is the same flag used for the delegate above, so both
    # sides of the pair are read the same way.
    if ! printf '%s\n' "${runtime_syms}" |
        grep -qE "(GLOBAL|WEAK)[[:space:]]+DEFAULT[[:space:]]+[0-9]+[[:space:]]+${register_backend}$"; then
        echo "FATAL: $(basename "${runtime}") does not export ${register_backend}, which ${target} imports." >&2
        echo "The delegate would fail to load with an undefined-symbol error." >&2
        echo "--- symbols matching register_backend in $(basename "${runtime}") ---" >&2
        printf '%s\n' "${runtime_syms}" | grep register_backend >&2 || echo "(none)" >&2
        exit 1
    fi

    # The ceiling is what the pinned ExecuTorch distribution already requires, and nothing more.
    # An earlier version raised it to a hardcoded manylinux_2_28 baseline to buy headroom, which
    # was unsound: this wheel is tagged bare linux_x86_64 (no auditwheel repair runs), and a bare
    # linux tag is in pip's compatible set on every x86-64 host regardless of glibc, so the wheel
    # promises nothing about the versions a host provides.
    #
    # Any host that can load ExecuTorch can load a delegate that stays within what ExecuTorch's own
    # libraries require, which is what makes this bound sound without a platform tag. It is not
    # generous: the widest family in the pinned wheel is GLIBCXX_3.4.22 (GCC 6), so std::filesystem
    # (3.4.26) is out of reach. Raising it further needs a tagged artifact to make the promise real:
    # build with --plat-name manylinux_2_28_x86_64 and require that prefix in CI, then the baseline
    # is a property of the wheel instead of an assumption about it. Use auditwheel's own numbers if
    # so; the ones guessed here were wrong in three of four families.
    # Compared against the manylinux platform the wheel ships under, not against the runtime's own
    # requirements. A symbol version requirement is a floor on the HOST, not a ceiling a library
    # imposes on its neighbours: two libraries in one process may need different GLIBCXX versions,
    # and the loader only needs the host's libstdc++ to satisfy the highest of them. Comparing
    # against the sibling wheel rejected the delegate wherever TensorRT was built with a newer
    # toolchain than ExecuTorch, and by that rule this check would reject TensorRT itself.
    #
    # The floor is a parameter because the two architectures ship under different tags. The x86_64
    # row builds in a manylinux_2_28 container, while the aarch64 row builds in manylinux_2_39
    # because TensorRT needs a newer glibc there than 2_28 provides; .github/scripts/filter-matrix.py
    # selects that container deliberately. Baking in one constant rejected the aarch64 row for
    # requiring exactly what its own platform guarantees.
    if [ -n "${glibc_floor}" ]; then
        for family in GLIBCXX CXXABI GLIBC GCC; do
            mine=$(highest "${target_versions}" "${family}")
            [ -n "${mine}" ] || continue
            allowed=$(floor_for "${family}" "${glibc_floor}")
            [ -n "${allowed}" ] || continue
            if [ "$(printf '%s\n%s\n' "${mine}" "${allowed}" |
                sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | tail -1)" != "${allowed}" ]; then
                echo "FATAL: ${target} requires ${family}_${mine}, above what ${glibc_floor} guarantees (${family}_${allowed})." >&2
                echo "The delegate would fail to load on a host that satisfies the platform tag it ships under." >&2
                echo "--- ${family} versions required by the delegate ---" >&2
                versions "${target_versions}" | grep "^${family}_" >&2
                exit 1
            fi
        done
    fi
fi

# The RUNPATH is what lets the delegate find its sibling distributions, so read it once and put it
# through a series of checks: that it exists, that it is DT_RUNPATH, that it carries every entry the
# build asked for, and that nothing absolute survived.
runpath=$(printf %s "${dyn}" |
    grep -E 'RUNPATH|RPATH' |
    sed 's/.*\[\(.*\)\]/\1/')
# Present at all: a delegate with no RUNPATH resolves libexecutorch.so only if the loader finds it
# some other way, which is exactly what this entry exists to guarantee it does not depend on.
if [ -z "${runpath}" ]; then
    fail "${target} carries no RUNPATH, so it would not find libexecutorch.so in the sibling executorch distribution"
fi
# DT_RUNPATH, not the older DT_RPATH. The pinned ExecuTorch passes --enable-new-dtags to get
# RUNPATH because RPATH is searched before LD_LIBRARY_PATH and applies transitively, so a consumer
# could not point a locally built runtime at their application. Both output dialects: binutils
# prints "(RPATH)" in parentheses, elfutils eu-readelf prints "RPATH" bare, and CMake accepts
# llvm-readelf by name, so a parenthesised-only pattern would let DT_RPATH through on the
# readers this script explicitly invites.
if printf %s "${dyn}" | grep -qE '[[:space:]]\(?RPATH\)?[[:space:]]'; then
    fail "${target} carries DT_RPATH rather than DT_RUNPATH, which the loader searches before LD_LIBRARY_PATH and applies to dependencies' dependencies"
fi
# Every entry the build asked for, in order, not one restated by hand. Spot-checking
# $ORIGIN/../../executorch/lib accepted a delegate with $ORIGIN/../../tensorrt_libs or
# $ORIGIN/../../nvidia/cu13/lib missing, both of which break on a user machine while passing here and
# in CI: the loader resolves those sonames out of ld.so.cache on any host that has a CUDA toolkit
# or another torch installed, so it cannot be used to test whether the RUNPATH carries them.
if [ -n "${expected_runpath}" ]; then
    if [ "${runpath}" != "${expected_runpath}" ]; then
        echo "FATAL: ${target} carries a RUNPATH the build did not ask for, so it may not reach the sibling distributions it needs on a machine where they are not findable another way:" >&2
        echo "  expected: ${expected_runpath}" >&2
        echo "  actual:   ${runpath}" >&2
        exit 1
    fi
elif ! printf '%s\n' "${runpath}" | tr ':' '\n' | grep -Fxq '$ORIGIN/../../executorch/lib'; then
    # Fallback for a direct invocation without the build's string.
    # -F, so the dots are literal. As a basic regex this matched $ORIGIN/xy/executorch/lib, and a
    # wrong depth also satisfies the absolute-entry check below, so it would have shipped.
    echo "FATAL: ${target} has a RUNPATH but not \$ORIGIN/../../executorch/lib, so it cannot reach the sibling executorch distribution:" >&2
    printf '%s\n' "${runpath}" | tr ':' '\n' | sed 's/^/  /' >&2
    exit 1
fi
# And nothing absolute may survive. ExecuTorch's imported targets contribute the build machine's
# own site-packages path as a raw link option, which would ship in the wheel and, because it is
# ordered ahead of the relative entries, would satisfy the loader on the build machine whatever
# the relative entries say. Stripping it is what makes a wrong depth observable rather than
# masked, so assert the strip actually happened.
absolute=$(printf %s "${runpath}" | tr ':' '\n' | grep -v '^\$ORIGIN' | grep -v '^$' || true)
if [ -n "${absolute}" ]; then
    echo "FATAL: ${target} carries RUNPATH entries that are not relative to the artifact, so they resolve against the build machine or the working directory rather than the wheel:" >&2
    # Quoted, and indented with sed rather than by word-splitting the list: an entry
    # containing a glob character would otherwise expand against the working directory and
    # report paths that are not in the RUNPATH at all.
    printf '%s\n' "${absolute}" | sed 's/^/  /' >&2
    exit 1
fi

exit 0
