# GitHub-first collaboration

Repository: https://github.com/JohnnyZPiggy/TorchTARK

## Roles and commands

| Role | Responsibility | Command |
| --- | --- | --- |
| Coordinator/user | Final project direction, priorities, authorization and veto; may revoke/narrow GPT merge authority; exclusive notebook creation/maintenance, runtime/Drive setup, actual CPU/GPU runs and result relay to GPT | Summons the relevant agent |
| GPT/technical lead | Mathematical/algorithmic specifications, task allocation, independent code/evidence review, acceptance/change requests and delegated technical PR merges; no notebook creation or edits | **检查 GitHub** |
| Codex/engineer | Authorized Python package, experiments, tests, scripts, benchmarks and CLI; feature-branch PR delivery; no notebook creation or edits | **推进 GitHub 任务** |

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
4. **Delegated GPT technical merge:** GPT may merge passing, independently
   reviewed PRs into `main` without per-PR coordinator approval. Before merging,
   inspect the latest PR head/diff, requirements, evidence and blockers. Do not
   merge incomplete, unsafe, out-of-scope or stale PRs. Distinguish CPU passing
   from pending real GPU validation. Report the PR URL and exact merge SHA;
   authorize subsequent tasks only after their prerequisite gate passes.
   The coordinator retains final direction, priorities and veto, and can revoke
   or narrow this authority at any time. Escalate material project-direction
   conflicts, destructive changes, security/access issues and unclear high-stakes
   decisions to the coordinator. Codex never pushes directly to `main` or merges
   PRs. Contract changes require a reviewed PR within delegated authority;
   changes to that authority require the coordinator's decision.
5. **Coordinator-owned Colab validation when required:** The coordinator alone
   writes and maintains a minimal notebook invoking tested Codex-maintained
   Python CLI/modules at a verified immutable commit with an explicit `--device`,
   without duplicating numerical kernels. The coordinator sets up the runtime
   and Drive destination, executes CPU/GPU experiments and shares real outputs,
   logs and manifests directly with GPT. GPT reviews numerical/device evidence
   and records confirmed findings and feedback on GitHub. Neither GPT nor Codex
   creates or edits notebooks; no notebook PR from either AI is required.
   Notebook creation alone does not prove GPU execution.

TASK-000 (#1 / PR #3) established the CPU smoke workflow. TASK-001 (#2) records
this protocol. TASK-002 (#4) is the coordinator-owned real-CUDA gate; its pending
GPU execution does not block explicitly authorized documentation. The earlier
notebook proposal PR #5 was closed without merge and is superseded by this
coordinator-exclusive ownership contract.

## Evidence and source of truth

- **Issues:** requirements, decisions, authorization, dependencies and blockers.
- **PRs:** changed files, implementation, commands/results, GPT review and revisions.
- **Commits:** exact reproducible source version. Include the full SHA in handoffs.
- **Google Drive / Colab:** coordinator-controlled notebooks, environment and
  persistent run artifacts. The coordinator shares actual outputs/logs/manifests
  directly with GPT; GPT records confirmed findings in the Issue/PR. Include
  commit SHA and a specific accessible run-folder reference when practical.
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
   changes, blocked, awaiting GPU validation and awaiting technical merge.
3. Give the coordinator a compact factual summary with links and unverified
   limitations; publish actionable review feedback in the relevant PR/Issue.
4. After the current gate passes, create or update exactly the next necessary
   Issue with scope, acceptance, priority, dependencies and owner. Avoid duplicates
   or contradictory assignments; explicitly gate authorization.
5. Own mathematical review and assess the coordinator's actual Colab outputs;
   never create or edit notebooks. Merge only after the review gate above and
   within current delegated authority; report the PR URL and exact merge SHA.
   Never claim unseen results or autonomous background work.

## Starting a new Codex session

Open the existing TorchTARK worktree in its current Dropbox location, with
GitHub access available. Ensure the reviewed root `AGENTS.md` is present in the
checkout and read it with this document. Before these documents are merged,
explicitly check out the reviewed workflow branch to load them. Then send
**推进 GitHub 任务** to execute one eligible task, or **检查 GitHub** for read-only
status. If the instructions or required access are unavailable, report the gap;
do not assume another agent's chat supplies them.

Never modify the excluded original reference archive/source or move/reorganize
the local worktree. The coordinator exclusively creates and maintains notebooks;
neither GPT nor Codex edits them. Ask the coordinator only when
approval, credentials, unavailable permissions/capabilities, conflicting
requirements or an important decision are required. Authentication or Git
configuration failures must be reported without workarounds.
