# GPU Keepalive Rules

- `gpu_keepalive.sh` should ALWAYS be running EXCEPT during benchmarks/profiling. It keeps the GPU allocated to this pod.
- Stop it right before `bench_serving` or profiling, start it right after.
- It MUST be running when launching sglang servers — without it the GPU gets reclaimed and the server subprocess dies silently.
- Script: `bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh start|stop`

# Task Checklists

- Check `tasks/*.md` files for unchecked `- [ ]` TODOs — these are usually overnight/long-running tasks.
- Main task file: `tasks/debug_kernel_server_perf.md` — persistent kernel v3 + LPT + mixed batch CUDA graphs.
- Perf model: `tasks/server_perf_model.md` — MLP-based latency prediction for prefill-decode colocation.
- Workload context: `tasks/realistic_workload_analysis.md` — dataset length stats, arrival rates, per-turn growth.
- Write results directly into the relevant `.md` file under the TODO bullet, not into separate files.
- When asked to fix/reinvestigate a bullet, you should **uncheck that box** to properly track the progress.
- When encountering problems in completing a bullet, you should try to fix it and **investigate every possible cause to the bottom**, instead of throwing a few hypotheses and checking that box. Strictly follow the task description and the user's intention.
- **IMPORTANT**: Do not stop on things like "This is getting very deep/I will need to dig deeper". Just **continue digging until you finish everything on the task list**. However, you are allowed to commit and push intermediate states.
- **Overnight mode**: Applies when the user says "overnight task", leaves you to work autonomously, OR when any pending task in the task list (`TaskList`) is tagged with `(overnight)` in its subject. In overnight mode, you MUST work through ALL pending tasks in sequence without waiting for user feedback. After finishing each task, immediately move to the next one. Only stop when all tasks are completed or you hit a hard blocker that truly requires user input (e.g., missing credentials, ambiguous requirements). Running out of ideas is NOT a valid reason to stop — try harder. Committing intermediate progress is NOT stopping — immediately continue working after every commit.
- **Task list discipline**: Always keep the task list (`TaskList`) up to date. Mark tasks `in_progress` when starting, `completed` when done. Check for pending tasks after completing each one. The task list is your TODO — if it has pending items, you have work to do.

# Implementation Standards

- Do NOT add fallbacks or shortcuts that disable a feature for "complex" cases. If a feature is requested, implement it fully. If a reference implementation exists (e.g., FA3), study it and follow the same approach rather than falling back to a simpler path.
- Do NOT add comments like "complex", "too hard", or "not supported yet" as justification for skipping functionality. Implement it or clearly explain what's blocking with a concrete TODO.
- Do NOT write self-reminder comments after fixing or deleting code (e.g., "// CG is orthogonal — ..."). Comments should explain the code, not document your thought process or lessons learned. Use memory files for that.

# Commits

- For every commit, document changes in `CHANGELOG.md` with version number updates. Focus on main functional changes and important fixes, not minor details.
