.. _torch_tensorrt_autotune_py:

torch_tensorrt.autotune
=======================

.. automodule:: torch_tensorrt.autotune

.. note::

   This module is experimental; its API may change.

Overview
--------

The Auto-Tuner looks for a faster way to compile and run your model with
Torch-TensorRT. It tries one change at a time, such as a lower precision, CUDA
graphs, another TensorRT builder setting, a generated kernel or an edit of
your own code. It measures every try in a fresh process and keeps a change
only when repeated measurements show a gain larger than the measured
run-to-run noise, with the outputs still within your accuracy bounds.

* **You provide** a workload (the model, representative inputs and how to
  check its outputs), a goal (lower latency or higher throughput), the
  accuracy and memory limits every result must keep, and a budget of trials
  and time.
* **You get** the winning config with its compiled artifact, a before/after
  report with profiles, and the record of every experiment. The winner is the
  best config found if it beats your starting point once more at the end;
  otherwise your starting point is kept.

**The loop in one paragraph.** A session first computes the reference
outputs of your model, measures your starting config (the *baseline*), and
measures how much the baseline's speed varies between fresh processes, which
sets the *noise threshold*. If some knob values generate region kernels, it
generates each one's kernels once, before the search (they are *prepared*),
and every later run that uses such a value must run exactly those kernels.
Then it repeats: it writes down everything
measured so far (the *planner context*); a *planner* reads it and proposes one
experiment, a change to the *best* config found so far; the session checks the
proposal against the options the workload allows (the *knobs*); and a fresh
worker process builds the *candidate*, checks its outputs and times it (the
*first measurement*). A candidate that beats the best config by more than the
noise threshold then runs *head-to-head* against it, in interleaved pairs of
fresh runs, and becomes the new best config if it wins (it is *accepted*).
When the budget is spent, the goal is reached or the planner has nothing left
to try, the best config faces the baseline once more (the *final check*), and
the winner is rebuilt, checked and exported with a report.

Inference runs the exported artifact; it needs no planner, LLM or session.

Glossary
^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Term
     - Meaning
   * - workload
     - Your model, its representative inputs, the knobs it honors and how its
       outputs are checked: a :class:`Workload`, or a :class:`ModuleWorkload`
       for a module with fixed inputs. A *workload factory*
       (:class:`WorkloadFactory`), a function with JSON arguments, builds it
       again in every worker process.
   * - candidate
     - One built config of the workload, ready to call and time (a
       :class:`Candidate`). Every measurement builds its own.
   * - knob
     - A named option with a short list of allowed values and a default, such
       as ``engine.precision`` with ``fp32``, ``fp16`` and ``bf16`` (a
       :class:`Knob`). Knobs and editable files are all a planner can change.
   * - config
     - The knob settings that differ from the knobs' defaults, such as
       ``{"engine.precision": "fp16"}``, plus an ``edit.<name>`` entry for
       each edited file. Equal configs have equal SHA-256 digests, so no
       config is measured twice.
   * - baseline
     - The config the session starts from: the workload's ``baseline()``
       (empty by default, so every knob at its default) with the session's
       ``baseline`` changes. It is trial ``0000``, and it is kept when nothing
       beats it.
   * - best
     - The best config so far. It starts as the baseline and is replaced each
       time a candidate is accepted. Every proposal is a change to it.
   * - trial
     - One numbered experiment, recorded in ``trials/NNNN/``: the planner's
       proposal, its check, its measurements and its decision.
   * - worker
     - A fresh Python process that runs one job, such as building and
       measuring one config, and writes its result to a file. Nothing is
       built or measured in the session's own process.
   * - qualify
     - A measurement qualifies when its worker succeeded, the outputs met the
       accuracy bounds both before and after timing, the GPU memory stayed
       within ``max_memory_mb`` and the timing was stable.
   * - planner
     - What chooses the next experiment: a :class:`RuleBasedPlanner`, which
       follows the measured bottleneck, or an :class:`LLMPlanner`, which asks
       an LLM coding agent. A planner only proposes; it cannot build, measure
       or accept anything.
   * - planner context
     - What the planner reads before each trial, saved as
       ``trials/NNNN/context.json``: the goal, noise, limits and budget, the
       baseline and the best config with their measurements, profiles and
       bottleneck hints, every earlier trial, the knobs, the editable files
       and the guidance.
   * - bottleneck hints
     - Named symptoms found in a measurement, such as ``launch_overhead``
       (the GPU waits for the host) or ``dominant_kernel`` (one kernel takes
       much of the GPU time), each with a strength from 0 to 1. Each knob
       names the hints it can relieve. Hints steer the search; they never
       decide whether a candidate is accepted.
   * - noise threshold
     - The smallest relative gain that counts as real:
       ``max(min_improvement, noise_factor * spread)``, where the spread is how
       much the baseline's metric moved across extra fresh-process runs of
       it (the *noise runs*).
   * - first measurement
     - A candidate's first measurement, made with a profile. The candidate
       must qualify and beat the best config by more than the noise
       threshold to go on to the head-to-head.
   * - head-to-head
     - Pairs of fresh runs of the best config and the candidate, back to
       back, alternating which runs first. The candidate wins when every run
       qualifies, every pair shows a gain and the median gain exceeds the
       noise threshold.
   * - accept
     - A candidate that wins its head-to-head is accepted: it becomes the
       best config. Every other measured candidate is rejected.
   * - final check
     - After the search, the best config faces the baseline in fresh pairs,
       by the head-to-head rule. If it does not pass, the baseline is kept.
   * - winner
     - The config the session ends with: the best config if it passed the
       final check, otherwise the baseline. The winner is exported and
       reported, and ``TuningResult.winner`` summarizes it.
   * - editable file
     - A source file the workload lets a planner edit between
       ``# EDITABLE-START`` and ``# EDITABLE-END`` lines (an
       :class:`EditableFile`). Each edited version is stored under its
       SHA-256 and selected in a config by ``edit.<name>``.
   * - region kernel
     - A kernel that a :class:`~torch_tensorrt.region.KernelGenerator`
       writes for one ``fused_kernel`` region of the model and caches, such as
       the kernel of a ``kernels.<region>`` knob's ``generated`` value.
   * - prepared value
     - A knob value listed in the knob's ``prepare``, such as
       ``kernels.<region>=generated``. The session builds it once, before the
       search, in a worker of its own, and records the region kernels that
       build generated (see `Generated kernels: prepare, pin, choose`_).
   * - pinned kernels
     - The kernel each generation task got in the preparations (its last
       one), by task digest. Every later run that uses a prepared value only
       loads kernels from the cache and must run, for each prepared task it
       has, that task's pinned kernel; otherwise it fails.
   * - region speedup
     - A region kernel's own speedup: the region's eager PyTorch or TensorRT
       time over the kernel's time, as kernel generation measured them on the
       region alone. It informs the planner; only whole-model measurements
       decide.
   * - guidance
     - Markdown text from you, kept as ``guidance.md``, that the LLM planner
       reads with every proposal.

Quick start
-----------

A session starts from a *workload factory*: a function that builds the model
and its representative inputs and returns a :class:`Workload`. For a module
with fixed example inputs, :class:`ModuleWorkload` is enough. The model below
stands in for yours:

.. code-block:: python

    # my_project/workloads.py
    import torch
    from torch_tensorrt.autotune import ModuleWorkload


    class TinyEncoder(torch.nn.Module):
        """A stack of residual MLP blocks."""

        def __init__(self, width: int, depth: int, approximate: str = "none") -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.LayerNorm(width),
                    torch.nn.Linear(width, 4 * width),
                    torch.nn.GELU(approximate=approximate),
                    torch.nn.Linear(4 * width, width),
                )
                for _ in range(depth)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = x + block(x)
            return x


    def create_workload(batch: int = 8, width: int = 1024, depth: int = 4) -> ModuleWorkload:
        torch.manual_seed(0)  # every worker builds the same weights and inputs
        model = TinyEncoder(width, depth).eval().cuda()
        inputs = [torch.randn(batch, 128, width, device="cuda")]
        return ModuleWorkload(
            model, inputs, name="tiny-encoder", compile_options={"min_block_size": 1}
        )

Run a session on it from Python:

.. code-block:: python

    from torch_tensorrt.autotune import (
        Budget,
        Goal,
        Limits,
        RuleBasedPlanner,
        TuningSession,
        WorkloadFactory,
    )

    session = TuningSession(
        WorkloadFactory("my_project/workloads.py:create_workload", {"batch": 8}),
        directory="sessions/tiny-encoder",
        goal=Goal(metric="latency_ms", min_improvement=0.02),
        limits=Limits(accuracy={"cosine_similarity": (">=", 0.999)}),
        budget=Budget(max_trials=10, max_time_s=3600),
        planner=RuleBasedPlanner(),
    )
    result = session.run()
    print(result.decision, result.config)
    print(result.report())

or from the command line (see `Command line`_):

.. code-block:: bash

    python -m torch_tensorrt.autotune run \
        --workload my_project/workloads.py:create_workload --args '{"batch": 8}' \
        --session sessions/tiny-encoder \
        --accuracy cosine_similarity '>=' 0.999 --max-trials 10 --max-time-s 3600

Both describe the same session, so either one resumes what the other started
(see `Resuming a session`_). The workload factory runs in every worker
process, so its arguments must be JSON data, it should be cheap to call, and
it must build the same weights and inputs every time (seed them, or load them
from files): the reference outputs are computed in one worker and compared
with the outputs of candidates built in others.

``session.run()`` returns a :class:`TuningResult`:

* ``decision``: ``"accepted"`` when a candidate beat the baseline in the final
  check, otherwise ``"baseline kept"``;
* ``config``: the winner's config, for example
  ``{"engine.precision": "fp16"}`` (``{}`` when the baseline was kept and has
  no changes);
* ``artifact``: the directory holding the exported winner
  (``sessions/tiny-encoder/artifact``), or ``None`` when there was no export
  or it did not pass;
* ``stop_reason``, ``baseline`` and ``winner`` (summaries of the baseline's
  and the winner's measurements), ``final_check``, ``noise`` and
  ``export_summary``;
* ``report()``: the text of ``report.md``.

.. _torch_tensorrt_autotune_steps:

How a session works
-------------------

This section follows one session from start to finish. Each step is
recorded in the session directory before the next one starts, so a session
that stops at any point can be resumed.

1. `Start or resume`_: create ``session.json``, or check that the directory
   holds this session.
2. `Reference outputs`_: a worker computes the outputs every candidate must
   reproduce, once per session.
3. `Baseline`_: a worker measures the starting config as trial ``0000``; it
   must qualify.
4. `Measure noise`_: workers measure the baseline again, which sets the
   noise threshold.
5. `Prepare kernels`_: for each prepared value, a worker builds it once and
   records the region kernels it generated.
6. The search loop, one trial at a time, until one of the
   `Stop conditions`_ holds:

   a. `Planner context`_: write down everything measured so far.
   b. `Ask the planner`_: get one proposal.
   c. `Check proposal`_: refuse a proposal that breaks the rules, before any
      worker runs.
   d. `First measurement`_: a worker measures the candidate, with a profile.
   e. `Head-to-head`_: pairs of workers measure the best config and the
      candidate, if the first measurement cleared the noise threshold.
   f. `Accept or reject`_: an accepted candidate becomes the best config.

7. `Final check`_: if a candidate was accepted, pairs of workers measure the
   baseline and the best config.
8. `Export and report`_: a worker rebuilds and exports the winner; the
   session writes its report.

Every measurement is one worker that builds its candidate from scratch, so
the cost of a session is easy to predict. A new session runs one reference
worker, the baseline, ``noise_runs`` noise runs, one preparation per prepared
value, one first measurement per measured trial (an invalid proposal runs
none) plus ``2 * head_to_head_pairs`` runs for each candidate that clears its
first measurement, ``2 * final_check_pairs`` runs for the final check if a
candidate was accepted, and one export. With the defaults, eight trials of
which none clears its first measurement take 13 workers; each head-to-head
adds 4, and so does the final check.

Start or resume
^^^^^^^^^^^^^^^

A session directory holds exactly one session, described by its
``session.json``:

* its identity: the workload factory and its arguments, the goal, the
  limits, the ``baseline`` changes and the ``timing``;
* the settings each run may change: the budget, the planner, the noise
  control, ``final_check_pairs``, ``export``, and the workers' Python
  executable and working directory;
* the time spent so far and, from the next step on, the pinned workload and
  environment;
* its format, ``torch_tensorrt.autotune session``, and the format's version.

A directory without ``session.json`` starts a new session. Otherwise the
recorded identity must equal this run's: if anything differs, the run stops
with a :class:`TuningError` that names it (``... holds a different session
(it differs in goal); use another directory``). A ``session.json`` of another
format or version is refused with a :class:`TuningError` too, and so is a
directory without one by ``resume`` (see `Command line`_).

When ``guidance`` is given, it is written to ``guidance.md`` if the file is
missing or the guidance differs from the guidance given last time; otherwise
the file, which you may have edited, is kept.

Reference outputs
^^^^^^^^^^^^^^^^^

A worker calls the workload's ``reference()`` once per session and saves the
outputs, moved to the CPU, as ``reference.pt``. They may be tensors, numbers
and containers of them. Every later measurement compares a candidate's
outputs with these. A resumed session keeps ``reference.pt`` and runs a
``describe`` worker instead.

Both workers also describe the workload and the environment, and the session
pins that description in ``session.json``:

* the workload: its name, knobs, baseline, timing, accuracy defaults and
  editable files (with each file's SHA-256), and the SHA-256 of the workload
  factory's file and of every ``.py`` file beside it;
* the environment: the Python, PyTorch, CUDA, TensorRT and Torch-TensorRT
  versions, the GPU and its driver, and the git commit and uncommitted
  changes of the Torch-TensorRT source (see `Pinned workload and
  environment`_).

If a later run describes anything differently, it stops with a
:class:`TuningError` that names up to eight differences, because the recorded
measurements and reference outputs no longer apply. If the worker fails, the
session stops with a :class:`TuningError` (``Preparing the workload
(reference) failed: ...``) that points at its ``worker.log``.

This step also fixes two things every measurement uses. The accuracy bounds
are the workload's ``accuracy_defaults()``, with the bounds of
``Limits.accuracy`` replacing them metric by metric. The timing protocol is
the session's ``timing`` or, when that is ``None``, the workload's
``timing()``.

Baseline
^^^^^^^^

The baseline config is the workload's ``baseline()`` with the session's
``baseline`` changes applied (``--baseline JSON`` on the command line). A
worker measures it once, with a profile, as trial ``0000``.

**Rule:** the baseline must qualify. Otherwise nothing could be compared with
it, and the session stops with a :class:`TuningError` (``The baseline is not
valid, so nothing can be compared with it: ...``). Its decision is recorded
as ``baseline`` or ``failed``. A resumed session reuses a baseline that
qualified and measures a failed one again, since the cause (a busy GPU, a
``trial_timeout_s`` since raised, a crash) may be gone.

Measure noise
^^^^^^^^^^^^^

The same config measured in two processes runs at different speeds: each
build can pick different TensorRT tactics, clocks drift and other jobs share
the GPU. These differences are usually much larger than the variation within
one run, so a gain of a few percent seen in one run proves nothing. The
session therefore measures the baseline again in ``NoiseControl.noise_runs``
more workers (2 by default) and computes:

* the values: the baseline's metric from the previous step and the metric of
  every noise run that qualifies;
* ``spread = (max(values) - min(values)) / median(values)``, when there are
  at least two values;
* ``noise threshold = max(Goal.min_improvement, NoiseControl.noise_factor * spread)``,
  with ``noise_factor`` 2 by default.

Without a spread, because ``noise_runs`` is 0 or no noise run qualified, the
noise threshold is ``Goal.min_improvement``. In the second case the session
warns, and the report says that the noise is unmeasured.

For example, with the baseline at 10.0 ms and noise runs at 10.2 ms and
9.9 ms, the spread is (10.2 - 9.9) / 10.0 = 3%, and the noise threshold is
max(2%, 2 x 3%) = 6%. From now on a candidate must beat the best config by
more than 6%.

The result is saved in ``noise/summary.json``. A resumed session reuses it
when it was made with the same ``noise_runs`` and, unless that is 0, measured
a spread.

Prepare kernels
^^^^^^^^^^^^^^^

Some knob values need a long build that every later build of them reuses,
such as a ``kernels.<region>`` knob's ``generated`` value, whose build
generates the region's kernel. A knob lists them in its ``prepare``. Before
the search, the session builds the baseline config with each such value, and
the settings it requires, in one worker per value (``kernels/<knob>=<value>``)
that may run for ``Budget.prepare_timeout_s`` (7200 s by default) and does no
timing, as long as the search would still run a trial. It records the region
kernels the build generated or loaded; every later run that uses a prepared
value must run, for each generation task they prepared, that task's kernel. A
value whose preparation failed cannot be proposed. `Generated kernels:
prepare, pin, choose`_ describes the step, the pinning and how the planners
use the prepared kernels.

Planner context
^^^^^^^^^^^^^^^

Each trial starts by writing ``trials/NNNN/context.json``, everything the
planner gets to decide the next experiment:

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Key
     - Content
   * - ``goal``
     - the metric, ``target`` and ``min_improvement``
   * - ``noise``
     - ``between_process_spread`` (the spread of `Measure noise`_),
       ``noise_threshold`` and ``head_to_head_pairs``
   * - ``limits``
     - the accuracy bounds in force and ``max_memory_mb``
   * - ``budget``
     - ``trials_used``, ``trials_left`` and ``time_left_s``
   * - ``baseline``, ``best``
     - each one's trial, config and status; for a measurement that
       succeeded, its metric, latency (median and quartiles), throughput,
       accuracy, memory and build time, otherwise the failed stage and
       error; and its structure, profile, bottleneck hints (in full,
       strongest first), rooflines and timing stability
       (``timing_stability``, see `Stable timing`_)
   * - ``history``
     - every trial after the baseline: its parent, ``changes``, the
       planner's own choices (``proposed``), rationale, status, decision and
       reason; for a measured trial, its metric, the names of its bottleneck
       hints and its head-to-head gains; for a failed worker, the stage,
       error, exit status and signal, and ``error_tail``: the start of its
       stack dump, or the end of its log
   * - ``knobs``
     - every knob with its values, default, description, bottleneck hints,
       requirements and ``current`` value in the best config
   * - ``tried``
     - the config of every trial that was measured, the baseline included
   * - ``editable_files``
     - only when the workload declares editable files: each file's name,
       path, description and SHA-256, the digest of the edited version the
       best config uses (``edited``, ``null`` for the original) and its
       fenced blocks as they are in the best config
   * - ``kernels``
     - only when the workload has prepared values: each one's knob, value,
       status and reason, the config it was prepared in, its build time, its
       smallest region speedup, whether the best config uses it, its warnings
       and its region kernels (see `Choose`_)
   * - ``guidance``
     - only when ``guidance.md`` exists: its first 20,000 characters

Ask the planner
^^^^^^^^^^^^^^^

The session calls ``planner.propose(context, workdir)``, with
``trials/NNNN/planner`` as the planner's work directory. The planner returns
a :class:`Proposal`: knob ``changes`` (knob name to new value), ``edits`` of
editable files and a ``rationale``; or ``stop=True`` with the reason in
``rationale``. ``proposal.json`` records it, with the name of the planner
that made it. `Planners`_ describes how each planner chooses.

A planner failure does not end the session. When the planner raises, returns
something that is not a :class:`Proposal`, or returns a malformed one
(changes that do not map knob names to values, edits that are not lists of
``{"search", "replace"}`` mappings, a rationale that is not text, anything
that is not JSON data), the error is recorded as ``planner_error`` in
``proposal.json`` and its traceback in ``planner/error.txt``, and a
:class:`RuleBasedPlanner` proposes this trial instead; the proposal's
``note`` says so. The next trial asks your planner again. If the rule-based
planner fails too, or has nothing left to try, the search stops.

Check proposal
^^^^^^^^^^^^^^

The session turns the proposal into the candidate's config, the best config
with the proposed changes, and refuses it if one of these checks fails, in
this order:

1. **Edits.** Each file's edits apply to the best config's version of the
   file (see `Code edits`_): every search text occurs exactly once inside
   the fenced blocks, the fence lines stay as they are, no block becomes
   empty, the edits change something but no more than 400 lines, and the
   edited file still compiles. An ``edit.<name>`` change must name a
   declared editable file and, unless it is empty, a stored edited file.
2. **Knobs.** The proposal changes something; every changed knob exists, and
   its new value is one of its allowed values; and every value in the new
   config has the settings it requires (see `Requirements`_). Settings at
   their default are dropped from the config.
3. **Prepared values.** The new config uses no prepared value whose
   preparation failed (``the config uses kernels.attn='generated', which
   could not be prepared (...)``).
4. **Novelty.** The new config differs from every config measured before,
   compared by SHA-256, rejected and failed ones included.

A refused proposal is recorded with the decision ``invalid`` and the reason
(``proposal rejected: ...``), and no worker runs. It does not use a trial of
the budget or count toward ``patience``, but three invalid proposals in a
row stop the search.

First measurement
^^^^^^^^^^^^^^^^^

The session writes ``trial.json``: the trial's number, its parent (the best
config's trial), the proposal, the new config, ``changes`` (every setting
that differs from the best config) and ``proposed`` (the planner's own
choices, without the settings that came along as requirements). Then one
worker measures the config, with a profile (see `How a worker measures a
candidate`_).

**Rule:** the candidate must qualify, and its gain over the best config must
exceed the noise threshold, where

* ``gain = (best - candidate) / best`` for ``latency_ms`` (the median latency
  per work unit) and ``(candidate - best) / best`` for ``throughput``;
* ``best`` is the best config's own first measurement (for the baseline, its
  measurement as trial ``0000``).

If either fails, the candidate is rejected with the reason, for example
``no gain beyond noise: latency_ms +3.1% (10 -> 9.69); needs more than
6.0%``. If both hold, it goes on to the head-to-head. With
``head_to_head_pairs=0`` there is no head-to-head: the first measurement
decides alone, and it must also show non-overlapping quartiles, the
candidate's 75th-percentile latency below the best config's 25th percentile
(compared as throughput when the two process different numbers of items per
work unit).

Continuing the example: the best config, still the baseline, ran at
10.0 ms. A candidate that runs at 9.0 ms gains (10.0 - 9.0) / 10.0 = 10%,
more than the 6% noise threshold, so it goes on to the head-to-head.

One case measures nothing: a config that uses a prepared value but needs a
region kernel that was never prepared, because one of its settings changes a
region's generation task (see `Pin`_), fails to build, since such runs only
load kernels from the cache. The trial is then recorded as ``invalid``, at
stage ``kernels``, with the region and task it misses: like a refused
proposal, it spends no trial and no patience, and the planner sees it in the
history.

Head-to-head
^^^^^^^^^^^^

Fresh workers measure the best config and the candidate again,
``NoiseControl.head_to_head_pairs`` times each (2 by default), in back-to-back
pairs that alternate which runs first, so that a slow drift (clocks warming
up, another job starting) cannot favor one side:

.. code-block:: text

    trials/0002/head_to_head/0-best        pair 0: the best config first
    trials/0002/head_to_head/0-candidate
    trials/0002/head_to_head/1-candidate   pair 1: the candidate first
    trials/0002/head_to_head/1-best

Each pair's gain is computed as for the first measurement, from the two runs
of that pair. **Rule:** the candidate wins when

* every run qualifies,
* every pair shows a gain (above zero), and
* the median of the pair gains exceeds the noise threshold.

Use an even number of pairs: with an odd number, the best config runs first
in the extra pair. Continuing the example, pair 0 measures the best config
at 10.0 ms and the candidate at 9.1 ms (a 9% gain), and pair 1 the candidate
at 9.3 ms and the best config at 10.0 ms (a 7% gain). Both pairs gain and
their median, 8%, exceeds 6%, so the candidate wins. Had the candidate run
slower than the best config in either pair, it would have lost, however
large its other gain.

Accept or reject
^^^^^^^^^^^^^^^^

``decision.json`` records the outcome, ``accepted`` or ``rejected``, with the
reason; after a head-to-head it also holds the first measurement's reason
(``first_measurement``) and the pairs with their gains (``head_to_head``).
Typical reasons are:

* ``won head-to-head: median paired gain +8.0% over 2 interleaved pairs
  (noise threshold 6.0%)``;
* ``lost head-to-head: gains +9.0%, -1.0% (every pair must gain and the
  median must exceed 6.0%)``;
* ``head-to-head failed: pair 1 candidate: accuracy: ...``, when a run did
  not qualify.

An accepted candidate becomes the best config: later proposals change it,
and later candidates are compared with its first measurement. A rejected one
leaves the best config as it was. Either way, the trial counts toward
``max_trials``, ``trials.tsv`` gets its line and ``session.json`` saves the
time spent, so a later run can resume from here.

Stop conditions
^^^^^^^^^^^^^^^

Before each trial, the session checks, in this order, whether

1. the best config reached ``Goal.target`` (``goal target reached``);
2. ``Budget.max_trials`` measured trials were used (``trial budget of 8
   used``);
3. ``Budget.max_time_s`` has passed, counting every run of the session
   (``time budget of 3600 s used``);
4. the last ``Budget.patience`` measured trials brought no accepted candidate
   (``no accepted candidate in the last 3 trials``);
5. the planner made three invalid proposals in a row (``the planner made 3
   invalid proposals in a row``).

The search also stops when the planner returns ``stop=True`` (``the planner
stopped: ...``), for example once the rule-based planner has tried every
option. The budget bounds the search, not the session: a trial that starts
in time finishes, and the final check and the export run after the search
stops. A resumed run starts a new count of invalid proposals.

Final check
^^^^^^^^^^^

Each accepted candidate was compared only with the best config of its time.
When the search stops, the best config therefore faces the baseline
directly: ``final_check_pairs`` (2 by default) fresh pairs, in
``final_check/0-baseline``, ``0-best``, ``1-best`` and ``1-baseline``, judged
by the head-to-head rule with the same noise threshold.

**Rule:** if the best config passes, it is the winner and the decision is
``accepted``; if not, the winner is the baseline and the decision is
``baseline kept``. When no candidate was accepted, there is no final check
and the baseline is kept. ``final_check/summary.json`` records the pairs. A
resumed session always measures the final check anew.

Export and report
^^^^^^^^^^^^^^^^^

Unless the session was created with ``export=False``, one more worker
rebuilds the winner (the baseline, if it was kept), checks its outputs
against the accuracy bounds before and after a short timing (one warm-up
call and three rounds of one call; only the accuracy is judged), and saves
it with the candidate's ``export()`` into ``export/files``. The rebuild is a
new engine, so it is checked like any candidate. Only if its outputs meet the
bounds do its files, with the winner's ``config.json``, replace
``artifact/``: the old directory is moved aside and deleted once the new one
is in place, so ``artifact/`` always holds exactly one export that passed.
The winner's config is also written to ``export/config.json``. The outcome,
recorded in ``export/summary.json``, ``TuningResult.export_summary`` and the
report, is one of:

* ``exported``: the rebuild met the bounds; ``TuningResult.artifact`` is the
  artifact directory;
* ``export_rejected``: the rebuild missed the bounds; its files stay in
  ``export/files`` for inspection;
* ``export_unsupported``: the workload cannot save the winner. A
  :meth:`Candidate.export` that raises ``NotImplementedError`` says so, and so
  does ``torch_tensorrt.save`` when it refuses an ``execute_in_torch`` region
  whose backend (such as Inductor) it cannot preserve. The session still
  finishes: rebuild the config instead;
* ``export_failed``: anything else, for example a failed rebuild or a
  timeout.

In the last three cases ``TuningResult.artifact`` is ``None``, and
``artifact/`` keeps the last export that passed, if there is one: the
summary's ``kept`` names it and the config it holds, and the report says
whether that is the winner's config.

Finally, the session writes:

* ``report.md``: the decision and stop reason; the goal, noise threshold,
  accuracy bounds, planner, environment and workload source; a before/after
  table (latency median and quartiles, throughput, accuracy, GPU memory,
  engines, partitions and plugins, build time and, when known, the share of
  bandwidth-bound speed); the final check; the winner's changes and full
  config; the artifact; when there are prepared values, a ``Kernels`` table
  of each one's status and region kernels (digest, origin, kernel and
  baseline time, region speedup, generation time) and whether the winner
  uses it; every trial; and the profiles of the baseline and the winner;
* ``report.json``: the same as data (``baseline`` and ``winner`` hold each
  one's trial, config and result, and ``kernels`` the prepared values, each
  with ``winner_uses``), with a summary of every trial;
* ``trials.tsv``: one line per decided trial, written after every decision,
  with the columns ``trial``, ``parent``, ``decision``, ``metric``,
  ``paired_gain`` (the head-to-head median gain), ``roofline`` (the share of
  bandwidth-bound speed), ``accuracy``, ``build_s``, ``changes``, ``reason``
  and, for a failed trial, ``stage`` and ``signal``.

Resuming a session
^^^^^^^^^^^^^^^^^^

Running the same session again in the same directory, with the same Python
call, the same ``run`` command, or ``python -m torch_tensorrt.autotune resume
DIR``, resumes it:

* the reference outputs, a baseline that qualified, the noise runs (when
  ``noise_runs`` is unchanged), every preparation that succeeded and every
  decided trial are reused;
* a trial interrupted before its decision was written is measured again, and
  so is a baseline that failed; a preparation that failed or was interrupted
  before its result is prepared again, as long as the search would still run
  a trial;
* the final check and the export always run anew.

A run may change the budget (``trial_timeout_s`` and ``prepare_timeout_s``
included), the planner, the
noise control, ``final_check_pairs`` and ``export``. It may not change the
workload factory or its arguments, the goal, the limits, the baseline changes
or the timing, and it is refused once the workload's code or the environment
changed (see `Reference outputs`_). Do not edit anything the workers import
while a session runs: the pins are checked when a run starts, so an edit made
during a run reaches its later workers unnoticed.

:meth:`TuningSession.resume` rebuilds a session from its ``session.json``,
with its recorded planner, goal, limits, noise control and other settings,
and optionally a new ``planner`` or ``budget``. The workers' ``env`` is not
recorded; set ``session.env`` again if the workers need it.

How a worker measures a candidate
---------------------------------

Every measurement, whether of the baseline, a noise run, a first
measurement, a head-to-head run, a final check run or the export, is one
worker process. The session writes the job to ``request.json`` and starts the
worker in its own process group:

.. code-block:: bash

    python -X faulthandler -P -m torch_tensorrt.autotune._worker trials/0001/request.json

The worker rebuilds the workload from the factory, gives every knob its value
under the config (defaults included), and writes ``result.json`` next to
``request.json``; its output goes to ``worker.log``. A worker that runs
longer than ``Budget.trial_timeout_s`` (1800 s by default) is stopped (see
`Failed workers`_). Inside the worker, the measurement runs in stages, and
the first stage that fails ends it, recording the stage and the error:

1. ``build``: ``workload.build(values)`` returns the candidate; the build
   time and the candidate's ``describe()`` (engines, partitions, plugins) are
   recorded.
2. ``setup``: ``candidate.setup()``, untimed, for example to enter CUDA-graph
   mode.
3. ``check``: one call, whose outputs are compared with the reference
   outputs (``accuracy``).
4. ``time``: the timing protocol below, until the timing is stable or the
   retries are used up.
5. ``check``: one more call after timing, compared again
   (``accuracy_after_timing``), so that outputs replayed from a captured CUDA
   graph are checked too.
6. ``profile``: for the baseline and first measurements only, a few traced
   calls (see `Profiles`_).
7. ``export``: in the export worker only.

Whatever happens, the worker then calls the candidate's ``close()`` and
derives the bottleneck hints. Its result also lists the region kernels the
build registered (``kernels``, see `Kernel records`_), and a run that uses a
prepared value is checked against its pinned kernels afterwards (see `Pin`_).

Timing protocol
^^^^^^^^^^^^^^^

One :class:`Timing` applies to every measurement of a session (only the
export uses a shorter one):

* ``warmup`` untimed calls (5 by default), then
* ``rounds`` samples (15 by default, at least 3). Each sample times
  ``calls`` back-to-back calls (3 by default) between two device
  synchronizations, one before the first call and one after the last, and
  divides the time by the number of work units they performed
  (``calls * candidate.units``).

The result's ``latency_ms`` holds the median, the quartiles (``p25`` and
``p75``), the minimum, maximum and mean of the samples, in milliseconds per
work unit. ``throughput`` is ``candidate.items * 1000 / median``, in items
per second. The goal's metric is the median latency or the throughput.

Stable timing
^^^^^^^^^^^^^

Before timing, the worker waits for the GPU to be idle: ``nvidia-smi`` must
report at most 5% utilization in two consecutive one-second samples. The
first attempt waits up to ``NoiseControl.quiet_gpu_s`` (600 s by default);
each retry waits what is left of it, but always looks for at least 3 s, so a
busy GPU is still seen. ``quiet_gpu_s=0`` turns the wait off, and with it the
busy check. When the utilization cannot be read (no ``nvidia-smi``), the
worker does not wait and does not count the GPU as busy.

The spread of an attempt is the interquartile range of its samples relative
to their median, ``(p75 - p25) / median``. An attempt whose spread exceeds
``NoiseControl.max_spread`` (3% by default), or that followed a wait in which
the GPU stayed busy, is timed again, after the wait and the warm-up calls, up
to ``NoiseControl.retries`` more times (2 by default). The timing is *stable*
when its last attempt's spread is within ``max_spread`` and the GPU was not
seen busy before it. The result's ``timing_stability`` records the attempts
(``attempts``, the ``spreads`` of each, ``gpu_wait_s``, ``gpu_quiet`` and
``stable``). An unstable measurement never qualifies. Keep the worst
case, ``quiet_gpu_s`` plus 3 s per retry plus the build and the timing, below
``trial_timeout_s``.

Accuracy
^^^^^^^^

``workload.compare(outputs, reference)`` returns accuracy metrics by name,
and every bound in force must hold, before and after timing. A bound on a
metric that ``compare`` did not report, or reported as NaN or infinite,
fails. The default ``compare`` is :func:`compare_outputs`, which compares
the output trees leaf by leaf and reports the worst leaf of each metric:

* floating-point leaves: ``max_abs_error``, ``rel_l2_error`` (the L2 norm of
  the difference over the reference's) and ``cosine_similarity``;
* integer and Boolean leaves: ``match_fraction``.

The default bounds (:meth:`Workload.accuracy_defaults`) are
``cosine_similarity >= 0.9999`` and ``rel_l2_error <= 0.01``.
``Limits.accuracy`` replaces them metric by metric and may add others; the
comparators are ``<=``, ``>=``, ``<`` and ``>``. Outputs with another
structure or shape than the reference fail the ``check`` stage.

Memory
^^^^^^

The worker records the GPU memory its process holds, from ``nvidia-smi``
(this includes TensorRT's own allocations), and PyTorch's peak allocated
memory from the first call on. ``Limits.max_memory_mb`` applies to the
first, or to the second when the process's memory cannot be read, for
example in a container whose process IDs differ from the host's; a
measurement with neither fails the limit.

Profiles
^^^^^^^^

The baseline and every first measurement trace ``Timing.profile_calls``
calls (3 by default) with the PyTorch profiler, after timing. Per work unit,
the profile reports:

* ``wall_ms``: the median latency from the timing; the traced calls run
  slower, so they are not used for it;
* ``device_ms``: the GPU time of every kernel, memory copy and memset;
* ``host_overhead_fraction``: ``1 - device_ms / wall_ms``, the share of the
  wall time in which the GPU waits for the host;
* ``kernels`` (launches), ``host_syncs`` (device-to-host memory copies) and
  ``copy_ms`` (time in memory copies and memsets);
* ``top_kernels``: the eight kernels with the most GPU time, with their
  share of it.

Spans that only annotate a range of device work, such as Inductor's
``## Call CompiledFxGraph`` ranges, are skipped, as PyTorch's own profiler
tables skip them.

Rooflines
^^^^^^^^^

A candidate can tell the worker how fast it could possibly run, so that the
report and the bottleneck hints can say how far it is from that floor:

* **Memory bandwidth:** set ``bytes_per_unit``, the bytes one work unit must
  read from device memory, such as its weights. The worker measures the
  GPU's peak read bandwidth once (a reduction over 512 MiB) and reports the
  floor, ``bytes_per_unit`` over that bandwidth, and the share of
  bandwidth-bound speed the candidate reaches, ``floor / median latency``.
* **Compute:** set ``flops_per_unit`` and ``peak_tflops`` (the best rate the
  GPU could run those operations at, for the candidate's precision), or let
  ``describe()`` report ``compute_floor_ms``, or ``gflop_per_unit`` with
  ``peak_tflops``. The floor is ``flops_per_unit / peak_tflops``. The worker
  does not measure a peak FLOP rate itself, since the right one depends on
  the precision.

Bottleneck hints
----------------

The worker derives a list of bottleneck hints from each measurement,
strongest first. A hint has a name, a strength from 0 to 1 and a detail line
that says what was measured. The rule-based planner tries the knobs that
name the best config's hints first; the LLM planner reads them all. No hint
decides whether a candidate is accepted.

.. list-table::
   :widths: 18 52 30
   :header-rows: 1

   * - Hint
     - Raised when
     - Strength
   * - ``roofline_gap``
     - the candidate reaches less than 80% of memory-bandwidth-bound speed
     - ``1 - share``
   * - ``compute_gap``
     - the candidate reaches less than 80% of compute-bound speed. When both
       floors are known, only the higher one, the one that binds, can raise
       its hint
     - ``1 - share``
   * - ``copy_overhead``
     - memory copies and memsets take at least 3% of the GPU time
     - 5 times that share, at most 1
   * - ``launch_overhead``
     - the GPU waits for the host at least 20% of the wall time
       (``host_overhead_fraction >= 0.2``)
     - that fraction
   * - ``host_syncs``
     - at least one device-to-host memory copy per work unit. Asynchronous
       copies count too, although they need not stall the host;
       synchronizations that copy nothing (stream, event or device) do not
     - ``0.5 + copies / 100``, at most 1
   * - ``dominant_kernel``
     - one kernel takes at least 35% of the GPU time
     - its share
   * - ``fallback``
     - the candidate has PyTorch partitions or more than one TensorRT engine
     - ``0.3 + 0.1 * (partitions + engines)``, at most 1
   * - ``precision``
     - ``engine.precision`` is ``fp32``
     - 0.4

Knobs
-----

Knobs are what a planner can change, besides editable files (see
`Code edits`_). A :class:`Knob` has a name ``<layer>.<option>``, its allowed
values (JSON scalars), a default, a description for the planner, the
bottleneck hints it can relieve and the settings of other knobs that its
values require. A config lists only the knobs whose value differs from the
default. When you declare a knob, list its values in the order a search
should try them and name the hints it relieves (``compute_gap`` for
compute-bound work): both planners rely on these declarations.

Layers
^^^^^^

Every knob belongs to one of five layers:

.. list-table::
   :widths: 14 40 46
   :header-rows: 1

   * - Layer
     - What it changes
     - Standard knobs
   * - ``graph``
     - how the graph is split between TensorRT and PyTorch
     - ``graph.min_block_size``
   * - ``engine``
     - how TensorRT builds its engines
     - ``engine.precision``, ``engine.optimization_level``,
       ``engine.tiling_optimization_level``
   * - ``kernels``
     - which regions run a generated kernel
     - ``kernels.<region>``, added by :class:`ModuleWorkload` ``regions``
   * - ``runtime``
     - how the compiled module runs
     - ``runtime.cudagraphs``, ``runtime.pre_allocated_outputs``
   * - ``workload``
     - options your workload defines, such as a model variant or a cache
       layout
     - none; declare your own, such as ``workload.gelu`` in
       `Writing a workload`_

Standard knobs
^^^^^^^^^^^^^^

:data:`STANDARD_KNOBS` are the knobs :class:`ModuleWorkload` honors, and the
default of :meth:`Workload.knobs`. Their values are listed in the order the
rule-based planner tries them; the first is the default.

.. list-table::
   :widths: 30 22 20 28
   :header-rows: 1

   * - Knob
     - Values
     - Relieves
     - In ``ModuleWorkload``
   * - ``runtime.cudagraphs``
     - ``False``, ``True``
     - ``launch_overhead``, ``host_syncs``
     - ``torch_tensorrt.runtime.enable_cudagraphs``
   * - ``runtime.pre_allocated_outputs``
     - ``False``, ``True``
     - ``launch_overhead``
     - ``torch_tensorrt.runtime.enable_pre_allocated_outputs``
   * - ``engine.precision``
     - ``fp32``, ``fp16``, ``bf16``
     - ``precision``, ``dominant_kernel``
     - Torch-TensorRT autocast to FP16 or BF16
   * - ``engine.optimization_level``
     - ``3``, ``5``, ``4``
     - ``dominant_kernel``
     - ``optimization_level``
   * - ``engine.tiling_optimization_level``
     - ``none``, ``moderate``, ``fast``, ``full``
     - ``dominant_kernel``
     - ``tiling_optimization_level``
   * - ``graph.min_block_size``
     - ``5``, ``1``, ``3``, ``10``
     - ``fallback``
     - ``min_block_size``

For :class:`ModuleWorkload`, the ``compile_options`` that a standard knob
controls (``min_block_size``, ``optimization_level`` and
``tiling_optimization_level``) become that knob's default, added to its
values if needed, so that the planner sees what actually runs.

``regions={"name": "submodule.path"}`` with ``generator=KernelGenerator(...)``
adds a ``kernels.name`` knob with the values ``native`` and ``generated``.
``generated`` compiles that submodule as a ``fused_kernel`` region named
``name``, whose kernel the :class:`~torch_tensorrt.region.KernelGenerator`
writes, validates and caches; such candidates compile with ``strict=True``.
``generated`` is a prepared value: the kernel search runs once, before the
search for configs, in a worker stopped after ``Budget.prepare_timeout_s``
(7200 s by default), and every later build loads the kernel from the
generator's cache, so the generator must keep ``use_cache=True`` (see
`Generated kernels: prepare, pin, choose`_). A ``generated`` value that your
``baseline`` sets is not prepared: the baseline's own worker generates its
kernel, and runs that use it are not pinned, so any of their workers may
search, within ``Budget.trial_timeout_s``. For such a region, give the
generator a ``timeout_s`` well below ``trial_timeout_s`` (``KernelGenerator``
allows 1800 s per region by default, as much as ``trial_timeout_s``).

Requirements
^^^^^^^^^^^^

A knob can declare ``requires``: settings of other knobs that its values
need (fused kernels that only run under CUDA graphs, say), in one of two
forms:

.. code-block:: python

    # Every non-default value needs CUDA graphs and FP16.
    Knob("kernels.fused_mlp", ("native", "fused"), "native", "...",
         requires={"runtime.cudagraphs": True, "engine.precision": "fp16"})

    # Each listed value needs its own settings; "padded" needs none extra.
    Knob("workload.layout", ("dense", "padded", "packed"), "dense", "...",
         requires={"packed": {"runtime.cudagraphs": True}})

The per-value form maps some of the knob's non-default values (never its
default, which a config omits) to the settings each needs; a sequence of
``(value, settings)`` pairs is accepted for values that are equal as
dictionary keys, such as ``1`` and ``True``. The rules:

* the session refuses a requirement on a knob or a value no knob offers, and
  a value whose requirements, followed transitively, ask for two values of
  one knob;
* a config that sets a value without everything that value needs is refused
  before anything is built, and so is a proposal that would create one (it
  is invalid and spends no trial);
* the rule-based planner proposes a value together with what it needs,
  followed transitively (a value that needs a setting that needs another
  brings both), and skips a value whose requirements cannot hold together
  with the best config's settings;
* the LLM planner must include missing requirements itself; they count
  toward its ``max_changes``.

A knob's ``to_json`` writes the per-value form as a list of
``{"value": ..., "requires": {...}}`` entries, so that values that are not
strings survive the session record; ``Knob.from_json`` reads both forms.

Generated kernels: prepare, pin, choose
---------------------------------------

Kernel generation is slow: a :class:`~torch_tensorrt.region.KernelGenerator`
may search for ``timeout_s`` (1800 s by default) per region, and a search
started by every worker that builds a config would cost each trial that long
and could give two runs of one config different kernels. So the session
generates each region kernel once, before the search (it *prepares* the knob
value that adds it), holds every later run that uses a prepared value to the
kernels the preparations produced (it *pins* them), and shows the planner
what each kernel achieved on its region, so that the planner can *choose*
between kernel sources.

Prepare
^^^^^^^

A knob lists the values whose build must be done once, before the search, in
``prepare``. The ``generated`` value of a :class:`ModuleWorkload` region knob
is one; a workload of your own declares its knobs the same way:

.. code-block:: python

    Knob("kernels.attn", ("native", "generated"), "native",
         "Generate a kernel for the attention region.", ("dominant_kernel",),
         prepare=("generated",))

``prepare`` lists non-default values of the knob, each once; another value is
refused when the knob is created. After the noise runs, for each prepared
value that the baseline config does not already set, one ``prepare`` worker
builds the baseline config with the value and the settings it requires (see
`Requirements`_). The worker:

* runs in ``kernels/<knob>=<value>/`` (``kernels/kernels.attn=generated/``),
  with its ``request.json``, ``worker.log`` and ``result.json``;
* is stopped after ``Budget.prepare_timeout_s`` (7200 s by default,
  ``--prepare-timeout-s``); leave room for the generator's ``timeout_s`` per
  region the value generates;
* builds the candidate and closes it, without checking or timing it;
* lists the region kernels the build registered, generated or loaded from the
  generator's cache, in its result's ``kernels`` (see `Kernel records`_).

Before each preparation, the session checks the conditions that stop the
search before a trial (see `Stop conditions`_), as they will stand once the
recorded trials are replayed: the goal target (a baseline that already
reaches it starts no preparation), ``max_trials``, ``max_time_s`` and
``patience``. Once one holds, no trial could use a new preparation, so the
remaining values are ``skipped``, except values that recorded trials use,
which the final check and the export need. Preparation time counts toward
``max_time_s`` like every other step, so a long preparation can use up the
time budget for the next one.

``kernels/summary.json`` lists every prepared value (``preparations``) with
its ``status`` (``ok``, ``failed`` or ``skipped``), the ``reason`` when it is
not ``ok``, the ``config`` it was prepared in, its ``build_s``, its
``regions``: the kernels of the generation tasks that the baseline's build
did not register, which are the regions the value adds or changes; its
``region_speedup``, the smallest region speedup among those kernels whose
speed was measured (``null`` when there is none); and its ``warnings``. It
also lists the ``pins`` (see `Pin`_) and the ``warnings``.

* A value whose preparation failed cannot be used: a proposal that uses it is
  invalid, with the reason, and spends no trial (see `Check proposal`_), and
  the rule-based planner never proposes it.
* A value that the baseline config sets is not prepared, since the baseline's
  own measurement built it; runs that use it are not pinned (see `Standard
  knobs`_ for the generator's ``timeout_s`` then).
* A resume reuses a preparation that succeeded with the same config. One that
  failed or was interrupted before its result is prepared again, like a
  failed baseline, since its cause (a crash, a busy GPU, a ``timeout_s`` or
  ``prepare_timeout_s`` since raised) may be gone, as long as the search would
  still run a trial. Delete a value's directory to prepare it again after it
  succeeded.

Pin
^^^

After the preparations, the kernels their builds registered, taken in the
order the preparations finished, give each generation task its pinned
kernel: the last kernel registered for it, which is the one the kernel cache
now holds. Generation tasks are the kernel cache key, so a region's name and
its index do not matter. ``kernels/summary.json`` lists each pin (``pins``:
``task_sha256``, ``region``, ``kernel_sha256`` and the ``values`` whose
builds registered the task). When a later preparation changed a task's
kernel, for example because an identical region's samples failed the cached
kernel and a new search replaced it, ``warnings`` says so and names every
value whose build registered that task; their ``regions`` then show the
final kernel (with ``replaced_kernel_sha256``, the one their own build
registered), and their runs are pinned to it.

Every later run whose config uses a prepared value, whether a first
measurement, a head-to-head run, a final check run or the export, is pinned:

* its ``request.json`` has ``"allow_generation": false``, and its worker
  builds with ``TORCH_TENSORRT_REGION_GENERATION=0``, so that every
  :class:`~torch_tensorrt.region.KernelGenerator` only loads cached kernels:
  a kernel missing from the cache fails the build at once instead of starting
  a search. With the switch off, a generator revalidates a cached kernel for
  correctness only: its speed was measured when it was accepted, so a busy
  GPU cannot fail the build;
* after the run, the session checks every kernel the run registered for a
  pinned task: it must be that task's pinned kernel. Kernels of tasks no
  preparation registered are not checked, and a pinned task that the run did
  not register is fine, since its config may not compile that region.

A run that ran another kernel for a pinned task fails at stage ``kernels``,
for example after the cached kernel changed:

.. code-block:: text

    kernels failed: The run did not use the prepared kernels: region 'attn#0'
    ran kernel 3f2a9c81d0e4 (cached) for generation task 77b0e1c2a9d3, not the
    kernel 0c41b7e2f6a8 that kernels.attn=generated prepared for it

A config can also change a region's generation task: kernel generation
records the engine build options it validates kernels with (such as
``engine.optimization_level`` and ``engine.tiling_optimization_level``), and
a precision change can change a region's data types. Such a config needs a
kernel that was never prepared, so its build fails (generation is off) and
measures nothing. The session then records the trial as an ``invalid``
proposal at stage ``kernels``, naming the region and the task it misses and
the configs its prepared values were prepared in: it spends no trial and no
patience, and the planner sees it in the history. Combine a prepared value
with settings like those it was prepared with.

Choose
^^^^^^

The planner context's ``kernels`` lists every prepared value:

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Key
     - Content
   * - ``knob``, ``value``
     - the prepared value
   * - ``status``, ``reason``
     - ``ok``, ``failed`` or ``skipped``, and why it is not ``ok``
   * - ``prepared_config``
     - the config the value was prepared in: the baseline with the value and
       what it requires
   * - ``build_s``
     - how long the preparation's build took, generation included
   * - ``region_speedup``
     - the smallest region speedup among its kernels whose speed was
       measured (``null`` when there is none)
   * - ``used_by_best``
     - whether the best config uses the value
   * - ``warnings``
     - the warnings about its tasks' kernels changing (see `Pin`_)
   * - ``regions``
     - the kernels of the regions it adds or changes, each a kernel record
       (see `Kernel records`_) with its task's pinned kernel, without its
       ``manifest`` path

The rule-based planner tries the prepared values whose kernels all run their
regions faster than their baselines (``region_speedup`` above 1) first,
fastest first, before any bottleneck hint's knobs and before any engine
option: they were prepared in the baseline config, and an accepted engine
option could change their regions' generation tasks and leave them unusable.
A prepared value whose kernels are slower waits for the layer order, and one
whose preparation failed is never proposed. The LLM planner's prompt explains
the section: prepared kernels are already measured per region, but a region
speedup is not a whole-model gain, so an experiment that uses one must still
win the whole-model head-to-head like any other, and a setting that changes a
region's generation task makes the experiment an invalid proposal. Either
way, only whole-model measurements accept a candidate.

Kernel records
^^^^^^^^^^^^^^

``torch_tensorrt.region.generated_kernels()`` returns the region kernels that
kernel generation registered in the current process, oldest first: one
``torch_tensorrt.region.GeneratedKernel`` per generated region of every
compilation. A worker reports those its job registered, as JSON, in its
result's ``kernels``. Each record has:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Field
     - Meaning
   * - ``region``, ``region_id``
     - the region's name, as given to ``fused_kernel(name=...)``, and its id
       in the compilation, ``<name>#<index>``
   * - ``task_sha256``
     - the digest of the generation task: the kernel cache key
   * - ``kernel_sha256``
     - the digest of the kernel, its source and launch settings, as its
       ``accepted.json`` records it
   * - ``backend``, ``provider``, ``model``
     - the kernel language, and the provider and model that wrote the kernel
   * - ``origin``
     - ``new`` (a search in this process accepted it), ``cached`` (a cached
       kernel passed its revalidation) or ``shared`` (an identical region of
       the same compilation registered it)
   * - ``generation_s``
     - the seconds the region's kernel took in this process: waiting for
       another process generating it, revalidation and any search
   * - ``latency_ms``, ``baseline``, ``baseline_latency_ms``, ``speedup``
     - the measurement the kernel was accepted with, from its
       ``accepted.json``: the kernel's median time on the region, the
       baseline it was compared with (``eager``, the captured PyTorch region,
       or ``tensorrt``, a TensorRT engine of the region under
       ``objective="tensorrt"``), the baseline's median time, and the region
       speedup, baseline time over kernel time
   * - ``manifest``
     - the ``accepted.json`` the provider and the measurement come from

A region whose kernel a later identical region of the same compilation
replaced shows the kernel it finally uses. Likewise,
``torch_tensorrt.region.missing_kernels()`` returns a ``MissingKernel``
(``region``, ``region_id``, ``task_sha256`` and ``reason``) for every region
whose kernel a compilation with generation off could not provide; a worker
reports those in its result's ``missing_kernels``. A workload that provides region kernels
in another way can record them with
``torch_tensorrt.region.record_generated_kernel`` and
``record_missing_kernel`` in its ``build``, and list the knob value that adds
them in ``prepare``, to have them prepared and pinned as well.

Writing a workload
------------------

:class:`ModuleWorkload` covers a module with fixed example inputs. It
compiles the module with ``torch_tensorrt.compile``, honors the standard
knobs, uses the module run eagerly (or ``reference_module``) as the reference
and exports with ``torch_tensorrt.save``. Its other arguments are ``name``,
``baseline`` (a config), ``timing``, ``knobs`` (instead of the standard
knobs), ``regions`` and ``generator``.

For anything else, such as state kept between calls, several work units per
call or options of your own, subclass :class:`Workload` and
:class:`Candidate`:

.. list-table::
   :widths: 28 28 44
   :header-rows: 1

   * - ``Workload`` member
     - Default
     - Purpose
   * - ``name``
     - ``"workload"``
     - the name shown in the records
   * - ``knobs()``
     - :data:`STANDARD_KNOBS`
     - the knobs the workload honors
   * - ``baseline()``
     - ``{}``
     - the config the session starts from
   * - ``build(values)``
     - required
     - build a :class:`Candidate`. ``values`` holds every declared knob's
       value, defaults included, and ``edit.<name>`` for each editable file:
       the path of the edited file, or ``None`` for the original
   * - ``reference()``
     - required
     - the outputs a correct candidate reproduces (tensors, numbers and
       containers of them), computed once per session
   * - ``compare(outputs, reference)``
     - :func:`compare_outputs`
     - accuracy metrics by name
   * - ``accuracy_defaults()``
     - ``cosine_similarity >= 0.9999``, ``rel_l2_error <= 0.01``
     - the bounds in force unless ``Limits.accuracy`` replaces them
   * - ``timing()``
     - ``Timing()``
     - the timing protocol
   * - ``editable_files()``
     - none
     - the files a planner may edit (see `Code edits`_)

.. list-table::
   :widths: 28 28 44
   :header-rows: 1

   * - ``Candidate`` member
     - Default
     - Purpose
   * - ``__call__()``
     - required
     - perform ``units`` work units of ``items`` items each and return the
       outputs that ``compare`` checks
   * - ``units``, ``items``
     - ``1``, ``1``
     - latency is reported per work unit and throughput as items per second
   * - ``setup()``
     - nothing
     - called once, untimed, before the first call
   * - ``describe()``
     - ``{}``
     - the candidate's structure; :func:`describe_module` counts a compiled
       module's TensorRT engines, PyTorch partitions and plugin regions
   * - ``export(directory)``
     - raises ``NotImplementedError``
     - save the candidate into ``directory`` and return the names of the
       files written
   * - ``close()``
     - nothing
     - release what ``setup()`` acquired; called after measuring
   * - ``bytes_per_unit``, ``flops_per_unit``, ``peak_tflops``
     - ``None``
     - the rooflines (see `Rooflines`_)

The workload below has an option of its own: the ``workload.gelu`` knob
switches every block's GELU to its cheaper tanh approximation, and the
accuracy bounds decide whether that is acceptable. It reuses ``TinyEncoder``
from the quick start:

.. code-block:: python

    # my_project/workloads.py (continued)
    import contextlib
    from pathlib import Path
    from typing import Any, Mapping

    import torch_tensorrt
    from torch_tensorrt.autotune import Candidate, Knob, Workload, describe_module

    KNOBS = (
        Knob("engine.precision", ("fp32", "fp16"), "fp32",
             "Precision of the TensorRT layers.", ("precision", "dominant_kernel")),
        Knob("runtime.cudagraphs", (False, True), False,
             "Replay the compiled module from a CUDA graph.", ("launch_overhead",)),
        Knob("workload.gelu", ("none", "tanh"), "none",
             "GELU formula of every block; 'tanh' is cheaper and less exact.",
             ("dominant_kernel",)),
    )


    class EncoderCandidate(Candidate):
        def __init__(self, compiled, inputs, cudagraphs: bool) -> None:
            self.compiled, self.inputs, self.cudagraphs = compiled, inputs, cudagraphs
            self.items = inputs[0].shape[0]  # sequences per call
            self._stack = contextlib.ExitStack()
            self._run = compiled

        def setup(self) -> None:
            if self.cudagraphs:
                self._run = self._stack.enter_context(
                    torch_tensorrt.runtime.enable_cudagraphs(self.compiled)
                )

        def __call__(self) -> torch.Tensor:
            with torch.inference_mode():
                return self._run(*self.inputs)

        def describe(self) -> dict[str, Any]:
            return describe_module(self.compiled)

        def export(self, directory: Path) -> list[str]:
            torch_tensorrt.save(
                self.compiled, str(directory / "model.ep"), arg_inputs=self.inputs
            )
            return ["model.ep"]

        def close(self) -> None:
            self._stack.close()


    class EncoderWorkload(Workload):
        name = "tiny-encoder"

        def __init__(self, batch: int, width: int, depth: int) -> None:
            self.batch, self.width, self.depth = batch, width, depth

        def knobs(self) -> tuple[Knob, ...]:
            return KNOBS

        def _model(self, approximate: str = "none") -> TinyEncoder:
            torch.manual_seed(0)
            return TinyEncoder(self.width, self.depth, approximate).eval().cuda()

        def _inputs(self) -> list[torch.Tensor]:
            torch.manual_seed(1)
            return [torch.randn(self.batch, 128, self.width, device="cuda")]

        def reference(self) -> torch.Tensor:
            with torch.inference_mode():
                return self._model()(*self._inputs())

        def build(self, values: Mapping[str, Any]) -> Candidate:
            inputs = self._inputs()
            precision = {}
            if values["engine.precision"] == "fp16":
                precision = {"enable_autocast": True,
                             "autocast_low_precision_type": torch.float16}
            compiled = torch_tensorrt.compile(
                self._model(values["workload.gelu"]), ir="dynamo",
                arg_inputs=inputs, min_block_size=1, **precision,
            )
            return EncoderCandidate(compiled, inputs, values["runtime.cudagraphs"])


    def create_encoder_workload(batch: int = 8, width: int = 1024, depth: int = 4):
        return EncoderWorkload(batch, width, depth)

Pass ``WorkloadFactory("my_project/workloads.py:create_encoder_workload")`` to
the session, or ``--workload my_project/workloads.py:create_encoder_workload``
to the command line. A candidate that cannot be saved raises
``NotImplementedError`` from ``export``, with the reason; the session then
records the export as unsupported (see `Export and report`_).

Code edits
----------

Knobs only reach what their authors anticipated; the largest gains often
need new code. A workload can declare :class:`EditableFile` objects for
source files it builds from, such as a lowering pass or region kernels.
Lines containing ``EDITABLE-START`` and ``EDITABLE-END`` fence the parts a
planner may edit:

.. code-block:: python

    # my_project/kernels.py
    # EDITABLE-START
    BLOCK_N = 8
    # EDITABLE-END

.. code-block:: python

    # my_project/workloads.py
    from pathlib import Path

    from torch_tensorrt.autotune import EditableFile, Workload, load_module

    KERNELS = Path(__file__).with_name("kernels.py")


    class MyWorkload(Workload):
        def editable_files(self):
            return [EditableFile("kernels", str(KERNELS),
                                 "Launch parameters of the fused kernel")]

        def build(self, values):
            # The edited file, or None for the original.
            path = values["edit.kernels"] or KERNELS
            kernels = load_module(path, "my_kernels")
            ...

A planner proposes SEARCH/REPLACE edits of a file (the LLM planner's reply
lists them as ``{"file", "search", "replace"}`` entries). The session applies
a file's edits in order, starting from the file as the best config has it,
each to the text the previous edit left, and refuses them unless

* each search text occurs exactly once inside the fenced blocks (it may also
  occur outside them; only the fenced occurrence is edited);
* the fence lines stay as they are, no replacement contains a fence, and no
  block becomes empty;
* the file's edits change at most 400 lines, counting for each edit the
  larger of its search and replace texts;
* the edits change the file, and the edited file still compiles.

The session stores the edited file under its SHA-256 in
``directory/edited_files`` and records ``edit.<name>`` in the config, so
edited candidates are reproducible, never measured twice and judged by the
same measurements and rules as knob changes. Edits that restore the original
file select the original. A proposal may also set ``edit.<name>`` among its
changes, to the digest of an edited file stored earlier or to ``""`` for the
original. The original file is part of the pinned workload, so editing it
between runs is detected like any other workload change.

A proposal cannot reach the measurement code, the reference or the accuracy
bounds: only fenced code changes. That code is not sandboxed, though. It
runs in the worker process that also checks the candidate's outputs and
times it, and the session only checks that it compiles. The fences are the
trust boundary: put them around code whose effect the measurement can check
(launch settings, kernel bodies), never around code that computes, copies or
compares outputs, and review the stored edited files before deploying an
edited config. The reference outputs are computed once, in a separate
worker, from the original code.

Planners
--------

A planner reads the planner context and returns one :class:`Proposal`. It
cannot build, measure or accept a candidate; the session checks and measures
everything it proposes.

Rule-based planner
^^^^^^^^^^^^^^^^^^

:class:`RuleBasedPlanner` is deterministic and needs nothing but the planner
context. It proposes one option, a knob and one of its values, at a time,
together with the settings that option requires. It goes through the knobs
in this order:

1. the prepared values whose region kernels all run faster than their
   baselines, fastest first (see `Choose`_);
2. for each of the best config's bottleneck hints, strongest first, the
   knobs that name that hint;
3. then every knob, layer by layer: ``runtime``, ``engine``, ``graph``,
   ``workload`` and finally ``kernels``.

Within a knob it tries the values in the order the knob lists them. It skips
the knob's current value, every option already tried (the baseline's values
and the options earlier trials chose), any option whose requirements cannot
hold together with the best config, any option that needs a prepared value
whose preparation failed, and any config already measured. Since the
baseline's values count as tried, it never proposes going back to them, so it
never reverts an accepted change. A setting that only came along as another
option's requirement does not count as tried: when ``kernels.k=on`` with the
CUDA graphs it requires is rejected, CUDA graphs alone remain a candidate
(each trial records the planner's own choice as ``proposed`` beside all its
``changes``). A proposal refused before it was built records no ``proposed``
choice, so every change it asked for counts as tried; one refused because its
build needed a kernel that was never prepared records its choice, which
counts.

When no untried option is left, it tries again the options that were only
tried under an earlier best config, because options interact (CUDA graphs
that did not pay under one workload option can pay under another). When
nothing is left at all, it stops with ``Every exposed option has been
tried``.

LLM planner
^^^^^^^^^^^

:class:`LLMPlanner` asks an LLM coding agent: the logged-in ``claude`` or
``codex`` CLI, or a trusted ``module:attribute`` adapter, as for
:class:`~torch_tensorrt.region.KernelGenerator`. Its prompt states the task
and the rules (at most ``max_changes`` knob changes to the best config, the
noise threshold a gain must exceed, how requirements work, not to repeat a
config) and appends the planner context as JSON, with the guidance in a
section of its own. The reply must follow a JSON schema:

.. code-block:: json

    {
      "action": "experiment",
      "changes": [{"knob": "runtime.cudagraphs", "value": true}],
      "rationale": "launch_overhead is the strongest bottleneck hint"
    }

``action`` is ``experiment`` or ``stop``. ``changes`` has at most
``max_changes`` entries (2 by default, at most 5); a reply with more is
refused even when the provider ignores the schema's limit. The LLM must
include the settings its changes require, and they count toward
``max_changes``. When the workload declares editable files, the reply also
has ``edits``: a list of ``{"file", "search", "replace"}`` entries, empty
when the experiment only changes knobs (see `Code edits`_).

When the provider fails in one of these ways, ``fallback`` (a
:class:`RuleBasedPlanner` by default) proposes this trial instead, the
proposal's ``note`` says why, and the next trial asks the LLM again:

* a ``ProviderError``, such as a CLI that exits with an error status;
* a ``ProviderUnavailableError``: a rate-limited or overloaded model API
  (HTTP 429 or 5xx);
* a timeout (``timeout_s``, 600 s by default);
* a malformed reply.

Any other exception, such as a ``codex`` turn that ends without writing a
reply, and any exception with ``fallback=None``, is a planner failure (see
`Ask the planner`_).

A ``module:attribute`` adapter is a callable, an object with a ``propose``
method or a class built without arguments. It is called, in a subprocess
stopped after ``timeout_s``, with the keyword arguments ``model``,
``prompt``, ``workdir``, ``timeout_s``, ``schema`` and ``evidence``, and
returns the reply as a JSON object. ``evidence`` is the planner context; the
provider layer, which kernel generation shares, calls it so.

Guidance
^^^^^^^^

``guidance`` (``--guidance FILE`` on the command line) is Markdown for the
LLM planner: what to try first, what to avoid, what you know about the
model. The session keeps it as ``guidance.md`` and reads the file again for
every proposal, so an edit, even during a run, steers the next proposal; the
file is replaced only when different guidance is passed. The LLM planner
shows it once in its prompt, together with its own ``instructions``. The
rule-based planner ignores it.

Writing a planner
^^^^^^^^^^^^^^^^^

Subclass :class:`Planner`, give it a ``name`` and implement
``propose(context, workdir)``, which returns a :class:`Proposal`. The session
checks whatever it proposes. :meth:`TuningSession.resume` and the ``resume``
command rebuild only the built-in planners; pass yours again with
``TuningSession.resume(directory, planner=...)``.

Failed workers
--------------

A worker that raises records the stage that failed and the exception. One
that dies without a result (a segfault in a native library, the
out-of-memory killer) or runs longer than its time limit (``trial_timeout_s``,
or ``prepare_timeout_s`` for a preparation) gets a result
written for it, with ``"synthesized": true``: stage ``crash`` or
``timeout``, its ``exit_status`` and the ``signal`` that ended it, and from
its log the first line that reports an error (``log_error``), the Python
stack dump the log ends with (``log_stack``) and the end of the log
(``log_tail``). A ``SIGKILL`` the session did not send is reported as a
possible out-of-memory kill.

Workers start with ``-X faulthandler``, so a fatal signal, even while they
import PyTorch or TensorRT, writes every thread's Python stack to
``worker.log``. When a worker exceeds its time, the session first sends it
``SIGUSR1``, on which the worker writes the same dump, and one second later
kills the worker's whole process group; the log then shows where it hung.
``faulthandler`` lists the main thread last and labels the thread that
received the signal ``Current thread``; ``log_stack`` starts with the thread
that explains the failure (the thread that faulted, or for a timeout the
main thread, which runs the job), innermost frame first, and then lists the
others. A worker that wrote its result and then exited abnormally keeps its
result, with how it ended added (and an ``exit_note`` if it ran past its
time limit).

A failed trial's reason, in ``decision.json``, ``trials.tsv``, the report
and the planner context, is its stage and the first line of its error (for
an exception, its ``Type: message`` line), with the signal and the first
error line of the log when there is one, for example ``crash failed: The
worker was killed by SIGSEGV (exit status -11) before writing a result; log:
Fatal Python error: Segmentation fault``. The planner context also carries
the start of ``log_stack``: the fatal error and the innermost frames of that
thread. Because the result is kept, a resumed session shows these trials as
failed, with their cause, and never runs them again.

Settings
--------

Every setting of a session, with the ``run`` option that sets it on the
command line (if there is one) and its default:

.. list-table::
   :widths: 28 26 14 32
   :header-rows: 1

   * - Setting
     - Command line
     - Default
     - Meaning
   * - :class:`WorkloadFactory`
     - ``--workload FACTORY``, ``--args JSON``
     - required, ``{}``
     - the workload factory, ``path/to/file.py:function`` or
       ``package.module:function``, and its keyword arguments
   * - ``TuningSession(directory=)``
     - ``--session DIR``
     - required
     - the session directory
   * - ``TuningSession(planner=)``
     - ``--planner``
     - ``rule-based``
     - :class:`RuleBasedPlanner` (``rule-based``) or :class:`LLMPlanner`
       (``llm``)
   * - ``LLMPlanner(provider=)``
     - ``--provider``
     - ``claude``
     - ``claude``, ``codex`` or a ``module:attribute`` adapter
   * - ``LLMPlanner(model=)``
     - ``--llm-model``
     - none
     - the model name passed to the LLM agent CLI
   * - ``LLMPlanner(timeout_s=, max_changes=, instructions=, fallback=)``
     - Python only
     - ``600``, ``2``, none, :class:`RuleBasedPlanner`
     - see `LLM planner`_
   * - ``TuningSession(guidance=)``
     - ``--guidance FILE``
     - none
     - Markdown guidance for the LLM planner (``guidance.md``)
   * - ``Goal.metric``
     - ``--metric``
     - ``latency_ms``
     - ``latency_ms`` (milliseconds per work unit, lower is better) or
       ``throughput`` (items per second, higher is better)
   * - ``Goal.min_improvement``
     - ``--min-improvement``
     - ``0.02``
     - the smallest relative gain that counts, in [0, 1); the noise
       threshold is never lower
   * - ``Goal.target``
     - ``--target``
     - none
     - stop the search once the best config reaches this value
   * - ``Limits.accuracy``
     - ``--accuracy METRIC OP VALUE``
     - the workload's
     - an accuracy bound, such as ``--accuracy cosine_similarity '>=' 0.999``
       (quote the comparator); repeat it for more metrics. Metrics not given
       keep the workload's default bounds
   * - ``Limits.max_memory_mb``
     - ``--max-memory-mb``
     - none
     - the most GPU memory a candidate may use
   * - ``Budget.max_trials``
     - ``--max-trials``
     - ``8``
     - the most measured trials; invalid proposals, the baseline and the
       final check do not count. ``0`` runs no trial: the session measures
       the baseline and the noise and exports the baseline
   * - ``Budget.max_time_s``
     - ``--max-time-s``
     - none
     - the wall-clock limit of the search, across runs
   * - ``Budget.patience``
     - ``--patience``
     - none
     - stop after this many measured trials in a row without an accepted
       candidate
   * - ``Budget.trial_timeout_s``
     - ``--trial-timeout-s``
     - ``1800``
     - stop any worker that runs longer, except a preparation
   * - ``Budget.prepare_timeout_s``
     - ``--prepare-timeout-s``
     - ``7200``
     - stop a worker that prepares a knob value (generates its region
       kernels) and runs longer
   * - ``TuningSession(baseline=)``
     - ``--baseline JSON``
     - none
     - knob changes to the workload's baseline config
   * - ``NoiseControl.noise_runs``
     - ``--noise-runs``
     - ``2``
     - extra runs of the baseline that measure the noise
   * - ``NoiseControl.noise_factor``
     - Python only
     - ``2.0``
     - the multiple of the spread in the noise threshold
   * - ``NoiseControl.head_to_head_pairs``
     - ``--head-to-head-pairs``
     - ``2``
     - best/candidate pairs a candidate must win; ``0`` decides on the first
       measurement alone
   * - ``TuningSession(final_check_pairs=)``
     - ``--final-check-pairs``
     - ``2``
     - baseline/best pairs of the final check; at least 1
   * - ``NoiseControl.quiet_gpu_s``
     - ``--quiet-gpu-s``
     - ``600``
     - how long each worker may wait for an idle GPU before timing; ``0``
       turns the wait off
   * - ``NoiseControl.max_spread``
     - ``--max-spread``
     - ``0.03``
     - the largest interquartile range, relative to the median, of a stable
       timing
   * - ``NoiseControl.retries``
     - ``--retries``
     - ``2``
     - how many times a worker times an unstable measurement again
   * - ``Timing`` (``warmup``, ``rounds``, ``calls``, ``profile_calls``)
     - Python only
     - ``5``, ``15``, ``3``, ``3``
     - the timing protocol; ``TuningSession(timing=)`` replaces the
       workload's ``timing()``
   * - ``TuningSession(export=)``
     - Python only
     - ``True``
     - rebuild, check and export the winner
   * - ``TuningSession(python=, env=)``
     - Python only
     - this Python, none
     - the workers' Python executable and extra environment variables

Command line
------------

``python -m torch_tensorrt.autotune`` runs, inspects and resumes sessions:

.. code-block:: bash

    python -m torch_tensorrt.autotune run --workload FACTORY --session DIR [options]
    python -m torch_tensorrt.autotune status DIR
    python -m torch_tensorrt.autotune report DIR
    python -m torch_tensorrt.autotune resume DIR [--max-trials N] [--max-time-s S] [--patience N]

* ``run`` creates the session its options describe (see `Settings`_), runs
  it and prints the report. Running the same command again resumes the
  session. An option value that the settings reject, such as a missing
  factory file or a negative budget, is reported as a usage error.
* ``status`` prints the workload, goal and planner, the time spent and one
  line per trial: its status (``ok``, ``failed``, ``invalid`` or
  ``running``), latency per work unit, decision, changes and reason. It only
  reads the session's records, so it also works while the session runs.
* ``report`` prints ``report.md`` once the session has finished, and exits
  with status 1 before.
* ``resume`` rebuilds the recorded session with
  :meth:`TuningSession.resume`, which keeps its planner, goal, limits and
  noise control, applies the given budget changes, runs it and prints the
  report.

For example:

.. code-block:: bash

    python -m torch_tensorrt.autotune status sessions/tiny-encoder
    python -m torch_tensorrt.autotune report sessions/tiny-encoder
    python -m torch_tensorrt.autotune resume sessions/tiny-encoder --max-trials 20

``status``, ``report`` and ``resume`` print an error and exit with status 1
for a directory that is not an Auto-Tuner session of this version.

The session directory
---------------------

.. code-block:: text

    sessions/tiny-encoder/
        session.json        identity, settings, time spent, pinned description
        reference.pt        the reference outputs
        reference/          the worker that computed them
        describe/           the worker that described the workload on resume
        trials/
            0000/           the baseline: trial.json, worker files, decision.json
            0001/
                context.json    the planner context
                planner/        the planner's work directory
                proposal.json   the proposal
                trial.json      parent, config, changes, proposed, proposal
                request.json    the first measurement's job,
                result.json     its result
                worker.log      and its output
                head_to_head/   0-best/, 0-candidate/, 1-candidate/, 1-best/
                decision.json   accepted, rejected or invalid, and why
        noise/              0/ and 1/ (the noise runs), summary.json
        kernels/            the prepared knob values:
            kernels.attn=generated/   its prepare worker's files
            summary.json    each value's status and kernels, the pins, warnings
        edited_files/       <sha256>.py for every edited file
        guidance.md         the guidance for the LLM planner
        final_check/        0-baseline/, 0-best/, 1-best/, 1-baseline/, summary.json
        export/             the export worker, config.json, summary.json, files/
        artifact/           the last export that passed, with its config.json
        trials.tsv          one line per decided trial
        report.md           the report
        report.json         the report as data
        events.jsonl        one JSON line per step

Each worker's directory holds its ``request.json``, ``result.json`` and
``worker.log``. An invalid proposal's trial directory has no worker files,
and ``planner/`` exists only when the planner used it (an LLM provider's
files, or ``error.txt``). ``session.json`` and the trial, worker and summary
records are written to a temporary file and renamed into place, so
``status`` never reads half a file.

Pinned workload and environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A directory holds one session: resuming with a different workload factory,
goal, limits, baseline changes or timing is refused, and so is resuming, or
reusing the reference outputs, once anything the measurements depend on
changed:

* the workload: its name, knobs, baseline, timing, accuracy defaults and
  editable files (with each file's SHA-256), and the SHA-256 of the workload
  factory's file and of every ``.py`` file beside it;
* the environment: the Python, PyTorch, CUDA, TensorRT and Torch-TensorRT
  versions, the GPU and its driver, and the Torch-TensorRT source: the git
  commit of its checkout and a SHA-256 of its uncommitted changes (``git
  diff HEAD`` of the package directory, untracked files included). Any new
  commit counts. The queries only read, and they read a checkout that
  another user owns too (git would refuse it as "dubious ownership"; the
  workers run its code anyway). An installed package that no checkout
  tracks has no source record (``null``); its version identifies it. When
  git cannot read the checkout (git is missing, the repository is damaged),
  the record is ``{"unavailable": <git's error>}`` and the session warns
  that a resume will not notice edits to the source.

The error names what changed. A directory that is not an Auto-Tuner session
of this version is refused with a :class:`TuningError`; start a new session
directory instead.

Limitations
-----------

* Candidates are measured on the workload you provide: the result is the
  best found within the budget for that workload, hardware and software, not
  a global optimum.
* Workers run on the machine and GPU of the session. They wait for the GPU to
  be idle before timing, but the session does not lock clocks; a GPU that
  stays busy longer than ``quiet_gpu_s`` makes measurements unstable, and an
  unstable baseline ends the session. Use an exclusive GPU, or set
  ``quiet_gpu_s=0`` and a looser ``max_spread`` on a GPU that is never idle
  (one driving a display, for example). Two sessions that share a GPU see
  each other's work as a busy GPU.
* Nothing locks a session directory: run one session in a directory at a
  time.
* Memory is the process's GPU memory as ``nvidia-smi`` reports it where the
  process ID is visible, otherwise PyTorch's peak allocation, which excludes
  TensorRT's own memory.
* The workload factory and an LLM adapter are trusted code. LLM replies are
  checked against the knobs, but knob implementations run as written, and
  edited code runs unsandboxed in the worker; see `Code edits`_.
* A prepared value is prepared once, in the baseline config. A config that
  combines it with a setting that changes a region's generation task (see
  `Pin`_) needs a kernel that was never prepared, so it is refused as an
  invalid proposal once its build fails; the session does not prepare such
  combinations.
* The artifact is what the workload's ``Candidate.export`` saves from a fresh
  build of the winning config, checked for accuracy but not timed against
  the final check. Runtime knobs (``runtime.*``, such as CUDA graphs) are
  execution modes, not part of a saved program: apply them when deploying,
  as ``config.json`` records.
* A program loaded with ``torch_tensorrt.load(path).module()`` checks every
  input on every call, which adds host time per call for engines with many
  inputs; ``module(check_guards=False)`` with the module's input-check
  pre-hook removed avoids it. With the Python runtime, a loaded program also
  runs its engines through the ``tensorrt::execute_engine`` operator, which
  copies every output that aliases an input; the compiled module run eagerly
  does not (see :ref:`auto_tuner`).

API reference
-------------

.. autoclass:: TuningSession
   :members: run, resume, check_result

.. autoclass:: TuningResult
   :members:

.. autoclass:: WorkloadFactory
   :members: load

.. autoclass:: Goal

.. autoclass:: Limits

.. autoclass:: Budget

.. autoclass:: Timing

.. autoclass:: NoiseControl

.. autoclass:: EditableFile

.. autofunction:: load_module

.. autoclass:: Workload
   :members:

.. autoclass:: Candidate
   :members:

.. autoclass:: ModuleWorkload

.. autoclass:: Knob

.. autodata:: STANDARD_KNOBS
   :no-value:

.. autoclass:: Planner
   :members: propose

.. autoclass:: RuleBasedPlanner

.. autoclass:: LLMPlanner

.. autoclass:: Proposal

.. autofunction:: compare_outputs

.. autofunction:: describe_module

.. autoexception:: TuningError
