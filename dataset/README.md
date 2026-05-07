# Dataset Notes

This directory is intentionally excluded from GitHub except for this note.

Keep raw datasets and locally generated JSONL files here during experiments, but do not commit them:

- `dataset/bigvul/`
- `dataset/devign/`
- `dataset/reveal/`
- `dataset/SARD/`
- `dataset/research_v1/`

Reason:

- several files exceed GitHub's file-size limits
- most files are generated artifacts that can be rebuilt locally

Recommended practice:

- keep large raw data outside the repository when possible
- use the scripts in `scripts/` to regenerate subsets and derived files as needed
