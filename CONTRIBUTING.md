# Contributing

Contributions should preserve the artifact's reproducibility and provenance contracts.

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Run `make check` (tests, method self-tests, and package build).
4. Keep credentials, corpora, generated indexes, outputs, and figures out of Git.
5. Record protocol changes explicitly; do not silently change model aliases, embedding
   models, candidate depths, denominators, or judge settings.

Bug reports should include the commit, Python version, relevant optional dependency
groups, a minimal reproduction, and sanitized logs.
