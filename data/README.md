# Data source and reproducibility

This project uses version 1 of [Kanchana1990's LLM Price-Performance Tracker](https://www.kaggle.com/datasets/kanchana1990/llm-price-performance-tracker-march-2026), a snapshot collected on March 31, 2026. The publisher attributes the data to Artificial Analysis, an Arena leaderboard API, and curated model metadata. These source claims have not been verified for every row.

As of October 28, 2026, the [official Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/kanchana1990/llm-price-performance-tracker-march-2026) lists only version 1, with a last update of March 31, 2026. The [complete file listing](https://www.kaggle.com/api/v1/datasets/list/kanchana1990/llm-price-performance-tracker-march-2026) contains one CSV and no later snapshot. This finding concerns this dataset, rather than all available LLM data.

## Download

From the repository root, run:

```bash
python src/download_data.py
```

The script downloads the [fixed version 1 archive](https://www.kaggle.com/api/v1/datasets/download/kanchana1990/llm-price-performance-tracker-march-2026?datasetVersionNumber=1), verifies its contents, and saves the CSV locally under `data/raw/`. The tested download did not require a Kaggle account. Raw source data is excluded from this public repository.

| Check | Expected value |
|---|---|
| Filename | `llm_price_performance_tracker_2026-03-31.csv` |
| Size | 99,892 bytes |
| Shape | 453 rows, 34 columns |
| Collection date | March 31, 2026 |
| Version | 1 |

CSV SHA-256:

```text
1c83338dc40f826ce19f3c6d17a3e03970979587fb860dccc79846f691e07177
```

The script stops if the checksum changes. It does not silently substitute a newer file. Source details and checksums are also recorded in `data/manifest.json`.

## Sample and variable definitions

The price screening reproduces the course sample:

| Screening result | Rows |
|---|---:|
| Original records | 453 |
| Missing blended price | 138 |
| Zero blended price | 16 |
| Positive-price records retained | 299 |

The outcome is `blended_cost_usd_per_1m`, expressed in USD per one million tokens. The analysis uses this blended price as supplied. Its exact input/output weighting has not been independently verified; it should not be described as an observed customer bill or a price per request.

The column named `provider` identifies the model creator or developer in the data card. It does not necessarily identify the inference API service selling access. Similarly, `is_open_source` represents the publisher's open-weight classification, rather than a separate audit of every model's license.

## Metadata limitations

Parameter counts and release years are curated fields without row-level citations in the supplied file. They have not all been independently verified. The data card also describes rule-based enrichment for the openness flag. These limitations do not establish that the whole dataset is synthetic, but they prevent treating every metadata value as an official measurement.

A source check found that Claude Opus 4.6 and Claude Sonnet 4.6 variants are assigned 2025 release years, while [Anthropic's official system-card index](https://www.anthropic.com/system-cards) dates both models to February 2026. The historical baseline preserves the original values. The extension includes a sensitivity specification excluding release year and parameter count, so those fields do not determine all reported conclusions.

## License and redistribution boundary

Kaggle labels this dataset **CC0: Public Domain**. That is the uploader's declared license; upstream redistribution rights have not been independently verified. [CC0](https://creativecommons.org/publicdomain/zero/1.0/) does not remove rights held by other parties or warrant the underlying work's copyright status.

This repository therefore provides the source link, verified download instructions, and analysis code rather than redistributing the raw CSV. It does not claim that all upstream rights have been cleared. Current Artificial Analysis API terms are not automatically treated as the contract governing this historical Kaggle download, and this limitation is not an allegation of infringement.
