# Clean-room statement

This portfolio project was written independently from public mathematical
descriptions of market-microstructure concepts. Earlier exploratory notebooks were
used only to identify broad topics worth demonstrating. Their source code, outputs,
data, credentials, organization-specific terminology, and trading rules were not
copied into this repository.

The implementation uses only deterministic synthetic data. The repository scanner
checks for common credential patterns, personal email addresses, local absolute paths,
and private-key headers in text and CSV files before publication. It cannot determine
whether a generic-looking idea is covered by an employer agreement; that still needs
the author's review.

Academic influences are documented in REFERENCES.md. Where a term can be confused
with a richer published model, the code uses a deliberately precise name. For example,
weighted_midprice is the static top-of-book weighted quote, not a reproduction of a
learned micro-price model.
