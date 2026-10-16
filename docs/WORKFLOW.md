# GitHub-first collaboration

Repository: https://github.com/JohnnyZPiggy/TorchTARK

## Roles and commands

| Role | Responsibility | Command |
| --- | --- | --- |
| Coordinator/user | Priorities, authorization, Colab runtime/Drive permissions, final merge decisions | Summons the relevant agent |
| GPT/project manager | Mathematical specifications, task allocation, independent review, all notebook authoring | **检查 GitHub** |
| Codex/engineer | Authorized Python source, experiments, tests and non-notebook utilities; feature-branch PR delivery | **推进 GitHub 任务** |

Codex **检查 GitHub** / **Check GitHub** alone means read-only inspection and
reporting unless execution is explicitly authorized. Codex's execution command
means complete one authorized, unblocked task in that invocation, then stop.
Prioritize outstanding PR revisions. Open or queued Issues are not permission
to start; read their latest authorization and dependency decisions.

## Handoff and gates

1. **Issue:** GPT records scope, acceptance criteria, priority, dependencies,
   owner and explicit authorization. The coordinator controls priorities.
2. **Codex PR:** Fetch latest state and read instructions, Issues, PRs and GPT
   reviews. Implement the selected task on a feature branch, run relevant
   checks, fix failures, push and open/update a PR with `Refs #N`. Publish exact
   evidence and blockers on GitHub. Stop after this task.
3. **GPT review:** Inspect the actual PR diff and accessible evidence against
   acceptance criteria. Write actionable feedback on GitHub. Codex handles
   authorized revisions on a later invocation. Waiting for review does not
   authorize unrelated work.
4. **Coordinator merge:** Only the coordinator decides whether to merge.
   Codex never pushes directly to `main` or merges PRs; GPT must not merge
   without explicit coordinator approval. Contract changes also require a PR
   and coordinator approval.
5. **Colab validation when required:** GPT authors a thin notebook invoking
   tested Python CLI/modules at a verified immutable commit, without duplicating
   numerical kernels. The coordinator starts the runtime, grants Drive access,
   supplies a real destination and runs it. GPT reviews the accessible artifacts.
   Notebook creation or merge alone does not prove GPU execution.

TASK-000 (#1 / PR #3) established the CPU smoke workflow. TASK-001 (#2) records
this protocol. TASK-002 (#4 / GPT-owned PR #5) is the separate real-CUDA gate;
its pending GPU execution does not block explicitly authorized documentation.

## Evidence and source of truth

- **Issues:** requirements, decisions, authorization, dependencies and blockers.
- **PRs:** changed files, implementation, commands/results, GPT review and revisions.
- **Commits:** exact reproducible source version. Include the full SHA in handoffs.
- **Google Drive:** persistent Colab datasets and run artifacts. Link the specific
  accessible run folder in the Issue/PR and include its commit SHA and manifest.
  A sync icon is not verification.

For each test/experiment, report exact commands, exit/results, environment and
hardware, missing dependencies, skipped tests and limitations. Distinguish
locally run tests from CI. Never report skipped CUDA tests as passed or silently
fall back to CPU. For Colab smoke runs retain `manifest.json`, stdout/stderr and
run metadata with the actual commit, device, PyTorch version, GPU model when
available, UTC timestamp and exit code. Label real CUDA execution as run,
skipped, unavailable or failed according to evidence; no speedup claims from
the smoke check.

Only repository-visible or connected accessible artifacts are reviewable.
Neither agent can see the other's private chat or unseen, unpushed local work.
Do not claim tests, artifact access or automatic monitoring without evidence.
Agents act when summoned, or through a separately configured supported
automation; connecting GitHub does not start background execution.

## GPT checklist for 检查 GitHub

1. Read open and relevant recently updated Issues/PRs, latest comments/reviews,
   commits/diffs, available CI status and linked accessible reports.
2. Compare code and evidence with acceptance criteria. Distinguish pass, needs
   changes, blocked, awaiting GPU validation and awaiting coordinator merge.
3. Give the coordinator a compact factual summary with links and unverified
   limitations; publish actionable review feedback in the relevant PR/Issue.
4. After the current gate passes, create or update exactly the next necessary
   Issue with scope, acceptance, priority, dependencies and owner. Avoid duplicates
   or contradictory assignments; explicitly gate authorization.
5. Own notebook changes and mathematical review. Never merge without approval
   or claim unseen results or autonomous background work.

## Starting a new Codex session

Open the existing TorchTARK worktree in its current Dropbox location, with
GitHub access available. Ensure the reviewed root `AGENTS.md` is present in the
checkout and read it with this document. Before these documents are merged,
explicitly check out the reviewed workflow branch to load them. Then send
**推进 GitHub 任务** to execute one eligible task, or **检查 GitHub** for read-only
status. If the instructions or required access are unavailable, report the gap;
do not assume another agent's chat supplies them.

Never modify the excluded original reference archive/source or move/reorganize
the local worktree. Codex never edits notebooks. Ask the coordinator only when
approval, credentials, unavailable permissions/capabilities, conflicting
requirements or an important decision are required. Authentication or Git
configuration failures must be reported without workarounds.
