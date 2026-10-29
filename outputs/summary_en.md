# LLM API Pricing: Reproduction and Extension

March 31, 2026 snapshot | Individual extension after the UW IS 451 course

## Question and data

Which observable model features are associated with paid API prices? Kaggle version 1 contains 453 rows: 138 missing prices and 16 zero prices leave 299 paid models. The outcome is blended USD per million tokens. The source's provider field identifies the model creator, not necessarily the inference API provider.

## Reproduction

Adding release year and applying the original one-pass p<0.05 selection naturally retains 13 predictors. Reduced R²=0.574544 and adjusted R²=0.555137 round to the report's 0.575 and 0.555. The no-year notebook version gives R²=0.566571. The report's main feature effects also match at its displayed precision.

## Diagnostics and validation

Singleton creator indicators produce leverage of one, making historical HC3 inference unavailable. The extension pools creators with fewer than five training rows, then applies HC3 and five grouped folds over 198 base-model groups. Imputation, selection and log-to-dollar correction use training rows only.

## Held-out errors (USD per million tokens)

Procedure | Log MAE | USD MAE | USD RMSE | OOF log R²
--- | --- | --- | --- | ---
Full extension | 0.902 | 3.800 | 16.394 | 0.409
One-pass selected | 0.927 | 3.601 | 16.054 | 0.371
Training means | 1.236 | 4.443 | 16.417 | -0.007

## Finding and business implication

The full extension's in-sample R² of 0.567 falls to 0.409 out of sample. Selection slightly lowers dollar errors but raises log errors. Large dollar errors remain close to the simple baseline, limiting the usefulness of fitted prices for budgeting.

## Price-deviation candidates

The main ranking covers 160 rows with seen creators, no imputed inputs and no known year error. Gemma 3n E4B Instruct and Nova Lite are below the prediction benchmark; o3-pro is above it. These are candidates for task-specific testing, not quality rankings, fair values or causal findings.

## Key limitations

One historical snapshot; 103 missing parameter counts and 84 missing release years. Five known year errors are preserved for reproduction. P-value selection is exploratory; family grouping is rule based and only one fixed fold assignment is used. No newer Kaggle snapshot was found, so no two-date comparison is reported.

Original course contribution: data screening from 453 to 299 rows. New code, diagnostics, validation and deviation analysis are the individual extension.

Source: https://www.kaggle.com/datasets/kanchana1990/llm-price-performance-tracker-march-2026 (v1, metadata checked 2026-10-28).
