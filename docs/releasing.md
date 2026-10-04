# Python release validation

The release workflow validates a version tag with `scripts/check-release.py --tag` before building artifacts. This strict mode accepts registry dependencies only; the publish job remains gated by the full reusable CI workflow and artifact verification.

During coordinated candidate validation, `python scripts/check-release.py --candidate` accepts the Python extension and engine only when both Cargo dependencies use the canonical engine repository, the same lowercase 40-character Git revision, and dependency versions matching the Python package version. Path dependencies, branches, and tags are rejected. The candidate mode is for validation only, not a publishable release.

CI runs `scripts/check-release.py --ci`. It validates either the approved paired Git candidate or normal registry dependencies and selects the matching fixture checkout: the exact candidate revision for Git dependencies, or `v<package-version>` for registry dependencies. The release workflow separately runs strict `--tag` validation, so CI candidate support does not permit Git dependencies in a published release.
