# Repository preparation checks

Checked locally on 20 November 2026:

- 61 CPU tests: 60 passed, 1 skipped because transformers was unavailable for the optional official-ALBERT test.
- Synthetic smoke test passed: training, source selection, save/load, prediction export, and baseline export.
- All 13 Slurm templates passed `bash -n`.
- The 23 original root Python implementations retain their original byte hashes.
- A temporary Git repository confirmed that Python bytes survive staging even with `core.autocrlf=true` because of `.gitattributes`.
- Actual Git ignore checks excluded representative weights, caches, data, predictions, logs, and environment files, while keeping synthetic examples and result exports.
- No private machine identifiers, model-weight files, oversized files, or missing local Markdown link targets were found in the source package.

GitHub-hosted Actions has not yet run, and GPU experiments were not rerun during repository preparation. These checks validate the source packaging and local CPU behavior; the recorded research results retain their original experimental scope.
