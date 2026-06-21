# IDE Evidence Report - Sprint 1.1

## 1. Files and Directories Inspected

Read-only inspection covered Git metadata, `.gitignore`, filename-only data safety checks, `latex_publication/v1`, `latex_publication/legacy`, and the top-level repository structure. No clinical-note content was opened or displayed.

## 2. Commands Run

```bash
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short
git remote -v
git log --oneline -n 10
git diff --check
git ls-files | sort
git ls-files | rg -i "mimic|physionet|note|notes|discharge|patient|subject|hadm|charttime|storetime|data/|raw|processed|json|csv|faiss|index|pkl|parquet|sqlite|db" || true
git log --all --name-only --pretty=format: | sort -u | rg -i "mimic|physionet|note|notes|discharge|patient|subject|hadm|charttime|storetime|data/|raw|processed|json|csv|faiss|index|pkl|parquet|sqlite|db" || true
git check-ignore -v data data/* data/**/* 2>/dev/null || true
git check-ignore -v "*.csv" "*.json" "*.index" "*.faiss" "*.pkl" "*.parquet" "*.sqlite" "*.db" 2>/dev/null || true
git status --short .gitignore
git diff -- .gitignore
git ls-files .gitignore
sed -n '1,220p' .gitignore
find latex_publication -maxdepth 4 -type f | sort
find latex_publication/v1 -maxdepth 3 -type f | sort
find latex_publication/legacy -maxdepth 4 -type f | sort
rg -n "\\documentclass|\\title|\\author|\\bibliography|\\addbibresource|\\input|\\include|\\begin\\{document\\}|\\end\\{document\\}" latex_publication/v1 || true
kpsewhich sn-jnl.cls || true
find latex_publication/v1 -maxdepth 2 -type f \( -name "*.pdf" -o -name "*.cls" -o -name "*.sty" -o -name "*.bst" \) -print | sort
find . -maxdepth 3 -type d | sort
find . -maxdepth 3 -type f \( -name "README.md" -o -name "*.md" -o -name "pyproject.toml" -o -name "*.toml" \) | sort
```

## 3. Git Identity and Status

- Current path and repository root: `/home/harry/Desktop/research/poc/clinical-rag-research`.
- Branch: `main`.
- Remote: `origin` fetch/push URL is `https://github.com/Harry5174/clinical-rag-research.git`.
- Working tree had a pre-existing modified `.gitignore`; diff validation was clean.
- Repository identity matches `clinical-rag-research`.

## 4. Remote / Public Alignment

The configured remote exactly matches `https://github.com/Harry5174/clinical-rag-research`.
The current `latex_publication/` tree is ignored and absent from tracked files, so the public Git repository does not currently contain the v1 paper-source freeze.

## 5. Public / Local Data Safety Findings

- No raw/restricted data-like path is tracked by the checked filename patterns.
- No raw/restricted data-like path appears in all Git-history filename checks.
- Tracked pattern matches are source/notebook filenames only: `notebooks/noteone.ipynb` and index-preparation source files.
- Local `data/raw`, `data/processed`, `data/vector_store`, JSON, CSV, pickle, and index assets are ignored.
- No restricted-data exposure stop condition was triggered.
- Future work should preserve filename-only safety checks and add a data-access note without adding data to Git.

## 6. .gitignore Findings

- `.gitignore` is tracked and currently modified.
- The local modification adds `review-docs/`; it was not changed in this sprint.
- It ignores raw/processed/vector data, CSV/JSON/pickle/index files, virtual environments, internal reports, plots, publication outputs, backups, and `latex_publication/`.
- It is sufficient for the visible restricted-data, index, and LaTeX-build categories. Result CSV/JSON files are globally ignored; tracked LaTeX result tables remain an intentional exception.
- Any future rule changes require a separately approved safety-focused sprint.

## 7. Paper Source Freeze Findings

- Current LaTeX source: `latex_publication/v1/sn-article.tex`.
- Current bibliography: `latex_publication/v1/sn-bibliography.bib`.
- `v1` appears to be the current source: it contains the manuscript, bibliography, and seven referenced figures.
- This source freeze is local-only until a future approved tracking/publication decision is made.
- No current PDF, `.cls`, `.sty`, or `.bst` file exists in `latex_publication/v1`.
- `kpsewhich sn-jnl.cls` found no installed class. The manuscript requires `sn-jnl` with `sn-mathphys-num`; compilation is not locally reproducible without that dependency.
- The current bibliography and figure references resolve by filename within `v1`; no paper compilation was attempted.

## 8. Legacy Structure Findings

`latex_publication/legacy/` contains prior v1 through v5 material, including the former multi-file draft and legacy Springer-source variants. Older versions are isolated from the current `latex_publication/v1` source as reported.

## 9. Repository Structure Findings

The repository separates application code, scripts, reports, results, visualizations, local data, review documents, and paper assets. It lacks a tracked `docs/research-revision/` area, data-access note, results manifest, supplementary-material package, and paper-build dependency record. This report establishes the first approved revision-documentation location.

## 10. Recommended Future Restructuring

Do not implement in this sprint. Future approved work should:

1. Keep `latex_publication/v1` frozen as the active paper source and `legacy/` read-only.
2. Use `docs/research-revision/` for decision logs, evidence reports, and revision plans.
3. Add metadata-only data-access guidance, results manifests, figure provenance, paper-build dependencies, and a supplementary-material manifest.
4. Keep restricted data and generated indexes outside Git.

## 11. Files Changed

`docs/research-revision/phase-01/sprint-1-1-repo-safety-and-source-freeze.md`

## 12. Research Integrity and Safety Confirmation

- No raw clinical-note text or raw MIMIC rows were exposed.
- No code, paper source, bibliography, results, or `.gitignore` content was edited.
- No destructive Git commands, commits, pushes, experiments, or LaTeX compilation were run.

## 13. Deviations or Scope Issues

The pre-existing `.gitignore` modification was preserved. The approved report file is the only Sprint 1.1 change. No restricted-data exposure was detected by the approved filename/path checks.

## 14. Recommendation

Ready for Implementation Supervisor review.
