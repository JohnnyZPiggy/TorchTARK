# TorchTARK: Codex instructions

Repository: https://github.com/JohnnyZPiggy/TorchTARK
Read [docs/WORKFLOW.md](docs/WORKFLOW.md) for team roles and evidence gates.

## Commands

- **检查 GitHub** (or **Check GitHub**) alone: inspect GitHub and report;
  read-only unless the coordinator explicitly authorizes execution.
- **推进 GitHub 任务**: execute exactly one authorized, unblocked engineering
  task through a feature-branch PR in the current invocation. Reading GitHub
  is the first step, not the completed task.

## Execution workflow

1. Run `git fetch origin`; inspect status, branch, commits, latest `origin/main`
   and applicable `AGENTS.md` instructions. Preserve unrelated local changes.
2. Read open Issues and PRs, their latest comments and GPT reviews, relevant
   diffs, and available CI/test evidence. Prefer requested revisions on existing
   PRs; otherwise select the highest-priority explicitly authorized, unblocked
   engineering assignment. An open Issue is not authorization. Latest explicit
   authorization may supersede an earlier queued state; never execute queued,
   blocked or unapproved work. If no eligible task exists, report why and stop.
3. Read the full acceptance criteria. For new work, branch from latest
   `origin/main`; for revisions, use the corresponding PR branch. Make the
   smallest complete change to Python source, experiments, tests or non-notebook
   utilities within the assignment.
4. Run relevant tests/experiments, fix failures and verify acceptance criteria.
   Report actual commands, results, environment/hardware, skip counts and
   limitations. Never fabricate tests or CI, or substitute CPU for requested CUDA.
5. Commit and push relevant changes to the feature branch. Create or update the
   corresponding PR targeting `main`, linking the Issue with `Refs #N`.
   Publish progress, exact commit SHA, test evidence, limitations and blockers
   on GitHub, not only in chat. Verify pushed SHA and final working-tree status.
6. Stop after this task for GPT review and coordinator merge. Do not begin
   another task until another instruction. When blocked or awaiting review or
   coordinator Colab execution, publish that state and stop.

## Boundaries

- Never create or modify notebooks (`notebooks/*.ipynb`); GPT owns them.
- Never push directly to `main` or merge PRs.
- Never modify the original author's excluded ZIP or extracted reference code.
- Keep the existing local worktree in its current Dropbox location; do not
  move, copy or reorganize it.
- Ask only for explicit approval, missing credentials/permissions or required
  capabilities, conflicting requirements, or a material decision. Stop and
  report authentication or Git configuration failures; do not use workarounds.
- Only accessible GitHub/connected artifacts can be reviewed. Do not claim
  access to another agent's private chat or unseen, unpushed local changes.
- Repository connectivity does not trigger background work. Execution requires
  a coordinator invocation or a separately configured supported automation.
- Changes to this collaboration contract require a PR and coordinator approval.
