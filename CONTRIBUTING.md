# Contributing

## Branches
- main: production, tagged releases only
- staging: release candidate
- dev: integration
- feat/<name>, data/<name>, exp/<member>-<idea>, fix/<name>

## Rules
- No direct pushes to dev, staging or main. Everything goes through a PR.
- Every PR needs 1 approval from a teammate and passing CI.
- Run dvc push before git push when data or models change.

## Commit messages
Conventional Commits: feat:, fix:, data:, exp:, docs:, chore:, build:, test:

## Merge strategy
PRs into dev are squash merged, to keep history clean.
