# Exclusions And Notes

## Included For Full Review

This package includes the reports, data files, code, result tables, visual outputs, saved model artifacts, metadata, and optional project workflow context.

## Excluded From Package

The following folders are intentionally not included:

- `.python_packages`
- `modeling/.python_packages`

These are machine-local installed dependency folders. They are large, not portfolio-friendly, and can be recreated from `03_Code_and_Reproducibility/requirements.txt`.

## Why This Is Still Complete For Recruiter Review

A recruiter or technical reviewer can inspect the project story, source data, modeling code, final reports, metrics, charts, predictions, model artifact, and reproducibility requirements without the local dependency folders.
