# Public repository file policy

## Included material

This repository contains model and analysis code, English technical documentation, source data snapshots, processed tables, trained checkpoints and saved research results. Earlier fusion stages remain dependencies of the final forecast. The official entry point is `m4_reports/final_strategy_manifest.json`.

Course submissions, member rosters, student identifiers, speaker assignments, registration spreadsheets, presentation/video production material and local audit evidence are not public source inputs. Do not commit those files or reintroduce private repository history into this release.

## Git LFS and exact bytes

Three large CSV tables use Git LFS: frozen distribution validation quantiles, frozen `fusion_v3` point-candidate forecasts and archived distribution test quantiles. Install Git LFS, then run:

```powershell
git lfs install --local
git lfs pull
git lfs fsck
```

Without Git LFS, these files may be small pointer text files instead of datasets. Code that reads them can fail or interpret the pointers incorrectly. A configured required LFS filter with a missing executable may prevent checkout. Installing Git LFS and running the commands above repairs an existing clone.

All other files use ordinary Git storage. `.gitattributes` preserves original line endings because frozen manifests validate exact SHA-256 hashes. Rebuild in a separate directory when changing model inputs.

## Local outputs and identity

`.gitignore` excludes credentials, environments, caches, archives, private presentation materials and local audits. Use a GitHub noreply email and a non-identifying author name for public commits. Review both file contents and author metadata before pushing. Private history and pull-request references must not be copied into a public release.

## Verification

```powershell
uv run python -B scripts/audit_submission_data.py
uv run python -B -m pytest -q m3_ml/tests/test_fusion_v6.py
```

The public export retains all frozen-model dependencies. Notebook outputs are cleared and two R output manifests use relative paths; neither change alters fitted parameters, predictions or evaluation metrics.
