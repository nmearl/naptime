# NAPTIME model artifacts

This directory tracks model metadata only. Large checkpoint files should not be
committed to the source repository.

Recommended deployment pattern:

- Store checkpoints as release assets, Zenodo artifacts, S3 objects, or Git LFS
  artifacts.
- Mount the selected checkpoint into the service container at
  `/models/latest.pt`, or set `NAPTIME_CHECKPOINT` to an absolute path.
- Keep `manifest.json` updated with the expected artifact names, checksums, and
  intended taxonomy.

