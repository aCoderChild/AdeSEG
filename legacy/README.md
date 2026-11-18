# Legacy and provenance-preserved experiments

This directory records the immutable historical code, but **does not duplicate
or mutate it**. Some checkpoint loaders verify the exact contents of
`modeling/*.py` using `__file__`. Moving those originals into this
directory would invalidate previously trained artifacts.

See [source-bound module manifest](original_source_bound_modules/README.md)
and [full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).

New experiments should import `adeseg.*`; historical runs should use
their original commit, scripts, data and pinned checkpoint hashes.
