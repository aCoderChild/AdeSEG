# Why Actions can fail before tests start

The repository has repeatedly returned `runner_id: 0`, `steps: []`, and
`404 BlobNotFound` for job logs. This means the workflow has **not reached**
checkout, dependency installation or Python testing.

A separate `runner_probe.yml` workflow runs only `echo` on
`ubuntu-latest` with no checkout. If this minimal job has zero steps as
well, changing the test suite or Python packages cannot resolve the blocker.

Owner-only checks:

1. Repository **Settings > Actions > General** — confirm Actions and
   GitHub-hosted runners are allowed.
2. Account **Settings > Billing and licensing > Budgets and alerts** —
   confirm Actions usage and payment/spending restrictions. Public standard
   hosted runners are normally free, but account-level billing restrictions
   can still block jobs.
3. Open a failing Actions run and inspect the *job annotation* (it is not
   exposed through the installed repository connector).
4. If billing and permissions are healthy, send the run URL and
   `runner_id=0` / no-step evidence to GitHub Support.

This repository does not and cannot change the account's billing/Actions
administration settings through its GitHub connection.

Local fallback while GitHub runner provisioning is blocked:

```bash
python -m pip install -r requirements.txt
python -m compileall -q adeseg datasets evaluation modeling scripts training tests
python -m pytest -q tests/
```

Tests and real-video inference must pass before reporting the cleanup as
fully verified. Historical checkpoint hashes and backups must be kept.
