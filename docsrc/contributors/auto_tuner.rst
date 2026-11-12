.. _auto_tuner:

Auto-Tuner
==========

``torch_tensorrt.autotune`` (see :ref:`torch_tensorrt_autotune_py`) searches a
workload's compiler configuration toward a measured goal. This page is for
contributors: where each part of a session lives in the code, and the rules
the implementation keeps. It uses the words of the user page's glossary
(workload, candidate, knob, config, baseline, best, trial, planner context,
bottleneck hints, noise threshold, first measurement, head-to-head, final
check, editable file, region kernel, prepared value, pinned kernels, region
speedup, guidance), and the code uses the same words: ``best``
is the best config so far, ``config`` the knob settings that differ from the
defaults, ``noise_threshold`` the gain a candidate must exceed. The one
exception is the provider layer shared with region kernel generation, whose
``propose`` takes the planner context as its ``evidence`` argument.

Module map
----------

The package is ``py/torch_tensorrt/autotune``:

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Module
     - What it owns
   * - ``__init__.py``
     - the public API, re-exported from the modules below
   * - ``_settings.py``
     - what a session is given: ``WorkloadFactory`` (a factory and its JSON
       arguments), ``Goal`` (with ``gain``), ``Limits`` (with ``bounds``),
       ``Budget`` (with ``trial_timeout_s`` and ``prepare_timeout_s``),
       ``Timing`` and ``NoiseControl``; ``violations``, the bounds a
       measurement misses; the ``Config`` type, and ``canonical`` and
       ``digest``, which give equal configs equal SHA-256 digests
   * - ``_workload.py``
     - the workload contract: ``Workload`` and ``Candidate``;
       ``ModuleWorkload`` for a module with fixed example inputs (standard
       knobs, generated-kernel regions whose ``generated`` value is
       prepared); ``compare_outputs`` (the default accuracy metrics) and
       ``describe_module`` (engines, partitions, plugin regions)
   * - ``_knobs.py``
     - ``Knob`` (with its ``requires`` and ``prepare``), ``STANDARD_KNOBS``,
       ``LAYERS`` and ``BOTTLENECK_HINTS``,
       and the config arithmetic: ``apply_changes`` (check and normalize a
       config), ``resolve`` (every knob's value under a config),
       ``difference``, ``with_requirements`` (a change plus what it needs,
       transitively), ``chosen_changes`` (a planner's own choices without
       the requirements they brought) and ``knobs_by_name`` (index the knobs
       and refuse requirements no config can meet); ``compile_settings`` and
       ``autocast_settings`` map standard knobs to compile options
   * - ``_code_edits.py``
     - ``EditableFile`` and its ``EDITABLE-START`` / ``EDITABLE-END``
       fences; ``apply_edits`` (SEARCH/REPLACE edits inside the fences),
       ``check_compiles``, ``EditedFileStore`` (edited files under their
       SHA-256) and ``load_module``
   * - ``_planner.py``
     - ``Proposal`` and ``Planner``; ``RuleBasedPlanner``, whose option order
       (``_options_in_order``) puts prepared values that beat their regions
       first; ``LLMPlanner`` (its prompt, with the prepared-kernel rules
       ``_KERNELS_PROMPT``, ``proposal_schema`` for the reply and
       ``_proposal_from_reply`` to parse it); ``check_proposal``, the knob
       check of a proposal; ``planner_from_json``, which rebuilds a
       built-in planner on resume
   * - ``_session.py``
     - ``TuningSession``, the controller: the session directory and its
       records, the budget, every step of a session (one method each, see
       below) and the acceptance rules; preparing knob values and pinning
       the runs that use them to their kernels; launching workers and writing
       the result of one that left none; ``TuningResult``, ``TuningError``
       and ``read_session_json``
   * - ``_worker.py``
     - the worker process, ``python -m torch_tensorrt.autotune._worker
       REQUEST.json``, and ``run_job``, which runs one of its four jobs:
       ``reference``, ``describe``, ``prepare`` or ``measure``, and lists the
       region kernels a ``prepare`` or ``measure`` job registered
       (``generation_allowed`` turns generation off for a pinned run); the
       description of the workload and the environment that the session pins
       (``describe_workload``, ``workload_files``, ``environment``,
       ``source_provenance``); ``write_json``, the atomic JSON writer
   * - ``_measure.py``
     - ``measure_candidate``: one measurement in a worker, stage by stage
       (build, setup, check, time, check, profile, export), with the
       quiet-GPU wait, the stability retries, memory and the rooflines;
       ``prepare_candidate``, the build alone, for a preparation
   * - ``_profile.py``
     - ``profile_calls`` and ``summarize_events`` (per-unit GPU time,
       launches, copies, top kernels) and ``bottleneck_hints``, with the
       thresholds that raise each hint
   * - ``_report.py``
     - ``report.md`` (with its ``Kernels`` table of prepared values,
       ``_kernel_lines``), ``report.json`` and ``trials.tsv``, and the
       wording of a failed result's reason (``failure_reason``) that every
       record uses
   * - ``__main__.py``
     - the command line: ``run``, ``status`` (which lists the prepared
       values), ``report`` and ``resume``

Kernel records come from the region API, outside the package:
``torch_tensorrt.region`` (``py/torch_tensorrt/region/_generation.py``)
defines ``GeneratedKernel``, ``generated_kernels`` and
``record_generated_kernel``, and ``MissingKernel``, ``missing_kernels`` and
``record_missing_kernel``. The generation controller
(``py/torch_tensorrt/dynamo/region_compilation/_generation/_factory.py``)
records every kernel it registers: ``_generate`` calls ``_record_kernel``
(new, cached or shared, with the measurement of the ``accepted.json`` that
accepted the kernel) and keeps the record in the compilation's memo
(``_Shared.records``), and ``final_registration`` calls
``_share_final_kernel`` when a later identical region's search replaced a
region's kernel, which replaces that compilation's own record by identity
(``_share_generated_kernel``), so concurrent or repeated compilations never
change each other's records. ``_Search.generate`` records a missing kernel
(``_Search.record_missing``) before it refuses a region that generation off
cannot serve, and ``_Search.from_cache`` revalidates a cached kernel for
correctness only when ``TORCH_TENSORRT_REGION_GENERATION=0`` blocked
generation (``_SWITCHED_OFF``). Recording never changes what generation does,
and a record that cannot be made or updated is logged, not raised.

Where each step lives
---------------------

``TuningSession.run`` drives a session, step by step, in ``_session.py``.
Each step that :ref:`How a session works <torch_tensorrt_autotune_steps>`
describes maps to methods there (or in the module named) and to the records
it writes:

.. list-table::
   :widths: 18 50 32
   :header-rows: 1

   * - Step
     - Code
     - Writes
   * - start or resume
     - ``_start_or_resume``: create ``session.json``, or check its format
       (``read_session_json``) and identity (``_identity``)
     - ``session.json``, ``guidance.md``
   * - reference outputs
     - ``_record_reference`` runs the ``reference`` or ``describe`` job;
       ``_pin_workload_and_environment`` records the description, or refuses
       a changed one (``_check_unchanged``)
     - ``reference.pt``, ``reference/``, ``describe/``
   * - baseline
     - ``_measure_baseline``, unless ``_is_validated_baseline`` finds a
       recorded one; ``check_result`` decides whether it qualifies
     - ``trials/0000/``
   * - measure noise
     - ``_measure_noise`` sets ``noise_spread`` and ``noise_threshold``
     - ``noise/``
   * - prepare kernels
     - ``_prepare_kernels``: a ``prepare`` job per prepared value
       (``_preparation_names``), unless ``_recorded_preparation`` holds a
       success or ``_stop_before_trials``, checked before each preparation,
       says the search runs no further trial; ``_preparation_entry`` keeps
       the kernels of tasks the baseline did not register; then
       ``_kernel_pins`` (each task's final kernel, in the order the
       preparations finished, ``_finished_at``, and the warnings) and
       ``_use_final_kernels``
     - ``kernels/<knob>=<value>/``, ``kernels/summary.json``
   * - pinning
     - ``_run_worker``, for every ``measure`` job whose config uses a
       prepared value (``_prepared_values``): the request's
       ``allow_generation``, then ``_pinned_kernel_problems`` against the
       session's pins (``_pins``), or ``_unprepared_reason`` for a build
       that missed kernels of tasks no preparation made; ``_decide`` makes
       that last case an ``invalid`` proposal
     - stage ``kernels`` in ``result.json``
   * - replay, on resume
     - ``_load_trials``, then ``_replay_recorded_trials``: decided trials are
       counted again, an interrupted one is measured and decided again
     - the missing ``decision.json``
   * - stop conditions
     - ``_stop_reason``, before each trial of ``_search``; ``_SearchState``
       holds the trials, the best config and the counts it checks
     - the stop reason in the report
   * - planner context
     - ``_planner_context``, with ``summary`` for the baseline and the best
       config, ``_history_entry`` for each earlier trial and
       ``_planner_kernels`` for each prepared value
     - ``context.json``
   * - ask the planner
     - ``_ask_planner`` (the planner, ``_malformed``, and the rule-based
       stand-in on a failure); ``_record_proposal``
     - ``proposal.json``, ``planner/error.txt``
   * - check proposal
     - ``_run_trial`` → ``_config_from_proposal``:
       ``_apply_proposed_edits`` (``_code_edits.apply_edits``,
       ``check_compiles``, ``EditedFileStore``), then
       ``_planner.check_proposal`` (``_knobs.apply_changes``), then the
       prepared values that failed (``_unusable_reason``), then the digest
       check
     - ``edited_files/``; for an invalid proposal, ``trial.json`` and
       ``decision.json``
   * - first measurement
     - ``_measure_trial`` → ``_measure_config`` → ``_run_worker`` →
       ``_run_worker_process``; in the worker, ``_worker.run_job`` →
       ``_measure.measure_candidate``
     - ``trial.json``, ``request.json``, ``result.json``, ``worker.log``
   * - head-to-head
     - ``_decide``: ``check_result`` against the best config, then
       ``_head_to_head``
     - ``head_to_head/``
   * - accept or reject
     - ``_decide`` returns the decision; ``_SearchState.count`` moves the
       best config and the counts; ``_search`` saves ``session.json``
       (``_Progress.checkpoint``)
     - ``decision.json``, ``trials.tsv``, ``session.json``
   * - final check
     - ``_final_check`` → ``_head_to_head``
     - ``final_check/``
   * - export and report
     - ``_export`` (``_replace_directory``, ``_restore_aside``,
       ``_kept_artifact``); ``_report.write_report`` and
       ``_report.write_trials_table``
     - ``export/``, ``artifact/``, ``report.md``, ``report.json``
   * - failed workers
     - ``_run_worker_process``, ``_kill_worker``, ``_exit_fields``,
       ``_read_worker_log``, ``_stack_excerpt``
     - a ``result.json`` with ``"synthesized": true``

``_log_event`` appends a line to ``events.jsonl`` for the main steps:
``reference`` or ``describe``, ``baseline_retry``, ``noise``, ``prepare``
and ``prepared`` (with its status and whether it was ``reused``),
``kernel_changed`` (a task whose kernel a later preparation changed),
``trial``,
``planner_failed``, ``stop`` (the planner stopped), ``decision``,
``final_check``, ``export`` and ``finished``.

The worker protocol
-------------------

The session and a worker share two JSON files in the worker's directory,
written with ``write_json``:

* ``request.json``: ``kind`` (``reference``, ``describe``, ``prepare`` or
  ``measure``), ``workload_factory``, ``reference_path``, ``result_path`` and
  ``session_dir`` (where ``edited_files/`` is); for ``prepare`` also
  ``config`` and ``timeout_s`` (the ``prepare_timeout_s`` it ran under); for
  ``measure`` also ``config``, ``timing``, ``profile``, ``noise_control``
  (``quiet_gpu_s``, ``retries``, ``max_spread``; the export sends none), for
  the export ``export_dir``, and for a pinned run ``"allow_generation":
  false``, which makes the worker build with
  ``TORCH_TENSORRT_REGION_GENERATION=0``.
* ``result.json``: ``status`` (``ok`` or ``failed``). A ``reference`` or
  ``describe`` result holds ``workload`` (the pinned description) and
  ``environment``, and a ``reference`` result also describes the saved
  outputs. A ``measure`` result holds ``build_s``, ``structure``,
  ``units``, ``items``, ``accuracy``, ``timing_stability`` (``attempts``,
  ``spreads``, ``gpu_wait_s``, ``gpu_quiet``, ``stable``),
  ``accuracy_after_timing``, ``latency_ms``, ``throughput``, ``samples_ms``,
  ``memory`` (``process_gpu_mb``, ``torch_peak_mb``), and when they apply
  ``roofline``, ``compute_roofline``, ``profile`` and ``exported``; then
  ``bottleneck_hints``. A ``prepare`` result holds ``build_s`` and
  ``structure``. A ``prepare`` or ``measure`` result, failed or not, also
  holds ``kernels``: the JSON of each ``GeneratedKernel`` the job
  registered, and ``missing_kernels`` when regions missed their kernel (each
  ``MissingKernel``). A failed result holds ``stage`` and ``error`` (and
  ``export_unsupported`` for the export stage). For a pinned run that ran
  another kernel for a pinned task, the session rewrites the result as
  failed at stage ``kernels``, keeping what was measured; for one whose
  build missed kernels of tasks no preparation made, it sets stage
  ``kernels``, keeps the build's error as ``build_error`` and lists those
  kernels as ``unprepared``. A result the
  session writes for a worker that left none adds ``exit_status``,
  ``signal``, ``log_error``, ``log_stack``, ``log_tail`` and
  ``synthesized``; one the worker wrote before it ended abnormally keeps its
  fields and gains ``exit_status``, ``signal`` and possibly ``exit_note``.

A worker reads only its request, ``reference.pt`` and the edited files its
config names; the session reads only a worker's ``result.json`` and, for a
worker that left no result, its ``worker.log``.

Design rules
------------

* **The planner only proposes.** A planner sees the planner context and
  returns a ``Proposal``; it never builds, measures or accepts. The session
  checks every proposal against the knobs and the editable files (unknown
  knobs, disallowed values, unmet requirements, edits outside the fences,
  empty changes and repeated configs are recorded as invalid), and only
  ``check_result`` and ``_head_to_head``, which read what a worker
  measured, accept a candidate. An LLM reply that is malformed or
  lists more than ``max_changes`` changes makes the LLM planner fall back. A
  planner that raises, or returns something that is not a well-formed
  ``Proposal`` (``_malformed``), is recorded, and a ``RuleBasedPlanner``
  proposes that trial; only when it has nothing left does the search stop.
* **One measurement per process.** Every job runs in a fresh worker started
  with ``python -X faulthandler -P`` in its own process group. Candidates
  cannot leak GPU memory, process-global lowering passes or converter
  registrations into each other, and a crash or a timeout (the process group
  is killed after ``trial_timeout_s``, or ``prepare_timeout_s`` for a
  preparation, following a ``SIGUSR1`` stack dump)
  becomes a failed trial whose ``result.json`` the session writes:
  ``exit_status``, ``signal``, the log's first error line and its final
  Python stack dump, reordered so the thread that faulted, or for a timeout
  the main thread, comes first (``_stack_excerpt``). Every reason names the
  first error line and the signal.
* **One measurement protocol.** Every candidate is checked on its first call
  and again after timing, timed with the session's one ``Timing`` (only the
  export, whose timing is not judged, uses a short one), and judged against
  the same bounds and the same noise threshold. Accuracy uses the reference
  outputs computed once per session. The export is a rebuild,
  so it is checked against the bounds too before its files reach
  ``artifact/``, which only an export that passed replaces (by renaming, so
  a failed export never destroys an earlier one); a winner the workload
  cannot save is recorded as ``export_unsupported`` and the session still
  finishes.
* **A gain must beat the measured noise.** Quartiles within one run are far
  tighter than the spread between processes running the same config (engine
  builds pick different tactics, clocks drift), so one run cannot show a
  gain of a few percent. The session measures that spread at the start (the
  noise runs), sets the noise threshold to a multiple of it, and runs any
  candidate that clears the threshold head-to-head in interleaved
  best/candidate worker pairs, all of which must gain. Workers wait for an
  idle GPU and time unstable samples again; an unstable measurement never
  qualifies. The winner faces the baseline again in the final check; a
  winner that does not hold up is dropped and the baseline is kept.
* **Generate once, then pin.** A slow build that later builds reuse, such as
  kernel generation, runs once per prepared value, in a worker of its own
  with its own time limit, never inside a measurement. Every later run that
  uses a prepared value only loads kernels and must register, for each
  prepared generation task, that task's pinned kernel, or it fails at stage
  ``kernels``; a run whose build needs a kernel no preparation made is an
  ``invalid`` proposal, at stage ``kernels``, not a measurement. So two runs
  of one config that uses prepared values measure the same kernels for their
  prepared tasks; runs of other configs (generated values the baseline sets,
  say) are not pinned and may generate. Region speedups from kernel
  generation only inform the planners; whole-model measurements alone accept
  a candidate.
* **Only fenced code changes.** Editable files let a planner edit source
  inside ``EDITABLE`` fences. Edits must match exactly once inside a fence, a
  file's edits may change at most 400 lines, the file must compile, and the
  result is stored by digest and named in the config. The measurement code,
  the reference, the accuracy bounds and the timing protocol are never
  editable through a proposal. Edited code is not sandboxed, though: it runs
  in the worker beside the output check, so the fences are the trust
  boundary and must not enclose code that computes or compares outputs.
* **Record before acting; resuming never skips a measurement.** Records are
  written atomically (``write_json``), and ``session.json`` is saved after
  every decision. It pins the session's format, identity, workload and
  environment (see the user guide's "Pinned workload and environment"); a
  run that differs in any of them is refused, and so is reusing the
  reference outputs. Completed trials are replayed from their
  ``decision.json``; a trial interrupted before its decision is measured
  again (an invalid proposal's ``trial.json`` already holds its decision, so
  a stop between its two writes loses nothing); a failed baseline is
  measured again; a preparation is reused once it finished (a failed one
  only under the same ``prepare_timeout_s``); and the final check always
  measures anew.
* **The budget bounds the search.** ``max_trials``, ``max_time_s`` and
  ``patience`` are checked before each trial; the final check and the export
  that follow are not cut short.
* **No silent fallbacks.** A planner failure, a provider fallback, a worker
  crash or timeout, noise that could not be measured, a source that cannot
  be pinned and an export that did not pass are all recorded with their
  reason, shown in the report and, where it helps the next proposal, in the
  planner context.

Adding knobs and hints
----------------------

Standard knobs belong in ``_knobs.STANDARD_KNOBS``, with a mapping in
``compile_settings`` or in ``ModuleWorkload``. Workload-specific options
belong in the workload's ``knobs()``, in the ``workload`` layer; the user
guide's "Knobs" section says how to order values and declare hints,
``requires`` and ``prepare``.

A knob's JSON is part of every pinned workload: changing a standard knob's
values, bottleneck hints, requirements or prepared values makes every
recorded session that uses it refuse to resume. Add a knob or a value
deliberately, and keep the JSON of the per-value ``requires`` form
unchanged.

A new bottleneck hint needs its name in ``_knobs.BOTTLENECK_HINTS`` and a
rule in ``_profile.bottleneck_hints`` that gives it a strength in [0, 1] and
a detail line saying what was measured; knobs that relieve it then name it.
Hints only order the search, so a hint must never be needed to accept a
candidate.

Tests
-----

The tests are in ``tests/py/dynamo/autotune`` (CI suite ``dynamo-autotune``):

* ``_workloads.py``: ``create_fake``, a CPU-only workload whose latency,
  accuracy and failures are scripted per knob value (``FAKE_KNOBS``), so
  session decisions are deterministic, and ``create_mlp``, a small module for
  GPU tests; ``_fake_editable.py`` is the fake workload's editable file. With
  ``kernels`` (or ``kernel_knob=True`` for ``kernels.fused`` alone) the fake
  workload has prepared knobs whose builds record fake region kernels with
  ``record_generated_kernel``, under region names and computations of your
  choice; their generation tasks depend on the optimization level, as engine
  build options are part of a real task. With ``kernel_cache`` it keeps a
  fake kernel cache, so that a build with generation off records a missing
  kernel and fails, as the generator does.
* ``test_session.py``: every step of a session and its records, prepared
  kernels and pinning included. Its ``jobs`` fixture runs worker jobs in the
  test process and can make runs slow, inaccurate or unstable.
* ``test_planner.py``: both planners, the prompt, the reply schema, the
  fallbacks and the use of prepared kernels.
* ``test_settings_and_knobs.py``: the settings' validation, knobs,
  requirements and prepared values, code edits, accuracy metrics, bottleneck
  hints, rooflines and source provenance.
* ``test_worker_failures.py``: real worker processes that crash, time out or
  stop before a decision, and the resume that follows; a preparation's own
  time limit, and a pinned worker that builds with generation off.
* ``test_cli.py``: the command line, including a real ``python -m
  torch_tensorrt.autotune run``.
* ``test_workers_and_modules.py``: real workers, and on a GPU
  ``ModuleWorkload``, an end-to-end session, region kernel generation and the
  roofline.

The kernel records are tested with the region API, in
``tests/py/dynamo/regions/test_generated_kernels.py``: a search, a cache hit
and identical regions record their kernels with the measurement of their
``accepted.json``; a region whose kernel a later identical region replaced
records the new one, and only its own compilation's records change; a build
with generation off records the kernel it misses, and under the environment
switch revalidates a cached kernel for correctness only; and a record that
cannot be made or updated never fails a compilation.
