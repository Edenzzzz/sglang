# GPU Keepalive Rules

- `gpu_keepalive.sh` should ALWAYS be running EXCEPT during benchmarks/profiling. It keeps the GPU allocated to this pod.
- Stop it right before `bench_serving` or profiling, start it right after.
- It MUST be running when launching sglang servers — without it the GPU gets reclaimed and the server subprocess dies silently.
- Script: `bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start|stop`

# Documentation Format

- Always keep task docs as a **structured, dated checklist**, so progress is
  visible at a glance rather than buried in prose.
- Each doc opens with a single `# Current goal` section at the top: what is being
  chased right now, and the tasks **in priority order**.
- Every checklist item carries a date and a one-line outcome. `[x]` means measured
  or validated against a reference; `[ ]` means not run. Never mark an item done
  because code was written.
- Condense and merge as findings supersede each other. A doc that accumulates
  every intermediate hypothesis is unreadable; keep the conclusion and the
  evidence, delete the path.
- **Group by item / feature, never chronologically.** Each task item is organised around the things the user
  asked to implement or try: the requirements as checkboxes, the current result, then one sub-section per
  feature (status, key numbers, command, commit). Do not append dated entries as work happens. Superseded runs
  are deleted or reduced to one line, and bugs found along the way go into one short list per item.

- **Fill the paper draft as experiments land.** `tasks/paper_outline.md` is the submission draft. Its story runs kernel
  -> single node -> multi node with SLO control, and each stage states its claimed contributions, the gains, and the
  metrics / graphs to show. When a result is measured, write it into its section at once (replacing that `[pending]`),
  so the draft is always a full paper with only the missing pieces open. Keep the claims consistent across sections:
  a result that changes a claim is reflected everywhere the claim appears (abstract, intro, the stage section).

- **The draft is written for human readers, towards a full high-quality conference submission (deadline 2026-10-30).**
  Every section is prose a reviewer can follow without the task docs. Define each term before using it; one idea per
  paragraph; no internal shorthand (build names, env flags, arm labels, capture-set names) outside the artifact
  appendix; numbers carry their unit and baseline. Lay out every stage (kernel -> single node -> multi node with SLO
  control) the same way: the claimed contribution, the gain, and the metric and figure that show it. Each figure is
  described in place (axes, arms, workload) with `[pending]` where the data is not in yet, so finishing the paper means
  only filling in measured pieces. After each result lands, reread the section it touches and keep the story
  coherent.

- **No lab bookkeeping in the paper.** The draft never mentions a failed experiment, a bug, a misconfiguration, a
  superseded run or a pending re-measurement **unless it clarifies why we landed on one of our core methods.**
  Trying to fuse FA2 and FA3 into one kernel and finding the warp layouts incompatible is worth keeping: it is the
  reason the design is two launches sharing one plan. "The baseline's rope base was misconfigured and those rows are
  being re-measured", "an earlier H20 pass gave a different number", "Table N supersedes these runs" are not --
  they tell a reader about our process, not about the system. Keep a result that justifies a design choice, in the
  present tense and as rationale rather than as version history; delete the rest. Validity caveats, superseded
  numbers and the audit trail live in the task docs, which is what they are for.

- **Exception -- the paper draft.** `tasks/paper_outline.md` is a paper draft, not a task doc. It follows the format
  of the user's original outline (`tasks/AttentionFuse Outline OSDI 25'.docx`): title, the outline bullets, Abstract,
  then sections 1-9 with the docx's subsection names. It has no `# Current goal`, no checklists and no dated
  TODO items; open work goes into the task docs, and pending results appear inline as `[pending]`.

# Reference Lists

- Every paper in a task doc's reference list gets exactly four bullets:
  1. **Summary:** the method and how it is evaluated (models, hardware, workloads, load axis, metrics, headline numbers).
  2. **Core method/contributions:** the mechanisms the paper introduces and claims as its contributions.
  3. **Relation to our method:** where it overlaps with or differs from ours.
  4. **What we can learn:** method or evaluation-setting lessons, used only to strengthen our core contributions:
     the fused kernels, higher cluster-wide request volume, and scheduling flexibility under load spikes, mixed
     workloads and workload variation. Do not stack the paper's techniques onto our system: that makes our
     contribution look like an assembly of others' ideas.
- Add every paper the user sends to the list in this format.

# Task Checklists

- Check `tasks/*.md` files for unchecked `- [ ]` TODOs — these are usually overnight/long-running tasks.
- Main task file: `tasks/debug_kernel_server_perf_4_mode.md` — FA3+FA2 split, 4-mode MTP step, serving A/Bs. (Predecessor, H100/H800 era: `tasks/archive/debug_kernel_server_perf.md`.)
- Multi-node scheduling: `tasks/pd_colocation_vs_disagg.md` — the four PD-colocation points.
- Index of all task docs: `tasks/README.md`.
- Perf model: `tasks/server_perf_model.md` — MLP-based latency prediction for prefill-decode colocation.
- Workload context: `tasks/realistic_workload_analysis.md` — dataset length stats, arrival rates, per-turn growth.
- Write results directly into the relevant `.md` file under the TODO bullet, not into separate files.
- When asked to fix/reinvestigate a bullet, you should **uncheck that box** to properly track the progress.
- When encountering problems in completing a bullet, you should try to fix it and **investigate every possible cause to the bottom**, instead of throwing a few hypotheses and checking that box. Strictly follow the task description and the user's intention.
- **IMPORTANT**: Do not stop on things like "This is getting very deep/I will need to dig deeper". Just **continue digging until you finish everything on the task list**. However, you are allowed to commit and push intermediate states.
- **Overnight mode**: Applies when the user says "overnight task", leaves you to work autonomously, OR when any pending task in the task list (`TaskList`) is tagged with `(overnight)` in its subject. In overnight mode, you MUST work through ALL pending tasks in sequence without waiting for user feedback. After finishing each task, immediately move to the next one. Only stop when all tasks are completed or you hit a hard blocker that truly requires user input (e.g., missing credentials, ambiguous requirements). Running out of ideas is NOT a valid reason to stop — try harder. Committing intermediate progress is NOT stopping — immediately continue working after every commit.
- **Task list discipline**: Always keep the task list (`TaskList`) up to date. Mark tasks `in_progress` when starting, `completed` when done. Check for pending tasks after completing each one. The task list is your TODO — if it has pending items, you have work to do.
- **No premature completion**: Do NOT mark tasks as completed when features are disabled, shortcuts are taken, or work is incomplete. "Code written but disabled" is NOT completed — debug it. "Using same model for both" when separate models are required is NOT completed. If something doesn't work, debug it fully instead of disabling it and moving on. A task is only completed when the feature is working end-to-end as specified.

# Implementation Standards

- Do NOT add fallbacks or shortcuts that disable a feature for "complex" cases. If a feature is requested, implement it fully. If a reference implementation exists (e.g., FA3), study it and follow the same approach rather than falling back to a simpler path.
- Do NOT add comments like "complex", "too hard", or "not supported yet" as justification for skipping functionality. Implement it or clearly explain what's blocking with a concrete TODO.
- Do NOT write self-reminder comments after fixing or deleting code (e.g., "// CG is orthogonal — ..."). Comments should explain the code, not document your thought process or lessons learned. Use memory files for that.
- **Never compromise by falling back silently**. If a feature (overlap scheduler, priority preemption, persistent kernel, etc.) crashes or misbehaves, DO NOT disable it to work around the problem. Find the root cause and fix it. Silent fallbacks mask real bugs and make benchmark comparisons apples-to-oranges. If you genuinely cannot fix it, stop and ask the user.

# Performance Results

- For ALL performance tables/numbers in task docs, always record: (1) the git commit ID (both sglang and flashinfer repos if applicable), and (2) the exact script/command used to produce the results. This makes results reproducible and enables bisecting regressions.

# Commits

- For every commit, document changes in `CHANGELOG.md` with version number updates. Focus on main functional changes and important fixes, not minor details.
