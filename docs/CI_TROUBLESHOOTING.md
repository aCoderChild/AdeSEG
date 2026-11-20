# GitHub Actions runner diagnosis

The user's actual GitHub Actions failure annotation states:

> The job was not started because your account is locked due to a billing issue.

This is an **account-level GitHub restriction**, not a Python code or dependency test failure. Recorded jobs had no assigned runner, no executed steps and zero billed execution milliseconds. Repository Actions permissions were enabled, and the visible Free-plan allowance was not exhausted.

The account owner must resolve the billing/authorization lock with GitHub billing support. Standard hosted runners normally do not require a paid GitHub plan for a public repository.

For local verification while hosted CI is unavailable:

```bash
python -m pip install -r requirements.txt
python -m compileall -q adeseg datasets evaluation modeling scripts training tests
python -m pytest -q tests
```

A complete CPU test suite and a small DEV model smoke run should precede a long end-to-end experiment. Re-running a blocked CI job cannot prove that new code passes.
