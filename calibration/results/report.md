# Global temperature calibration of the final UniMC models

## Experimental protocol

The final v0.3 decision-layer release was evaluated at repository commit `b170b3b83df1758dcd7e752a210750f5ca1d2624`. Original UniMC and four decision-layer architectures were included. Each decision layer had training seeds 0, 1 and 2.

One positive global temperature was fitted for original UniMC, and one for each trained decision-layer seed, giving 13 temperatures. Fitting minimized example-mean NLL on the pooled source calibration set of 3,193 questions (ARC-C 112, CommonsenseQA 974, OpenBookQA 496, PIQA 1,611). Every calibration question had equal weight. The fitted temperature was then shared across evaluation datasets for that model and seed. Target labels were used solely for evaluation.

For valid option logits z, calibrated probabilities were softmax(z / T). Optimization used float64 arithmetic and bisection of the convex NLL derivative in inverse temperature beta = 1/T, constrained to [0.0001, 10000]. Every fitted solution was interior.

ECE used maximum-probability confidence and 15 equal-width, right-closed bins. NLL used natural logarithms. Multiclass Brier was the example mean of the sum of squared probability errors over valid options, using the [0, 2] convention for every task. Padding contributed zero probability. CSV files store ECE and Accuracy as fractions.

Target summaries first averaged the four task-level metrics equally across SST-2, CB, COPA and HellaSwag, then reported the mean and sample standard deviation across training seeds (ddof=1). Original UniMC was evaluated once after verifying that its logits were identical in all twelve prediction copies. Domain-test and additional official-test results are available separately in metrics.csv.

## Target-task macro results

ECE is reported in percent. Values after ± are sample standard deviations across three seeds.

| Model | Stage | ECE (%) | NLL | Brier | Accuracy (%) |
|---|---|---:|---:|---:|---:|
| Original UniMC | before | 8.9864 | 0.5446 | 0.3065 | 77.5261 |
| Original UniMC | after | 9.3873 | 0.5501 | 0.3078 | 77.5261 |
| Residual MLP | before | 10.5439 ± 1.2337 | 0.6501 ± 0.0300 | 0.3650 ± 0.0152 | 73.3689 ± 1.0756 |
| Residual MLP | after | 10.4475 ± 0.3409 | 0.6432 ± 0.0243 | 0.3629 ± 0.0137 | 73.3689 ± 1.0756 |
| Transformer without interaction | before | 12.1163 ± 0.3433 | 0.6181 ± 0.0211 | 0.3431 ± 0.0106 | 75.3587 ± 0.8573 |
| Transformer without interaction | after | 11.3047 ± 0.3218 | 0.6117 ± 0.0199 | 0.3419 ± 0.0105 | 75.3587 ± 0.8573 |
| Transformer with interaction | before | 11.0361 ± 0.4748 | 0.6253 ± 0.0121 | 0.3478 ± 0.0067 | 75.0971 ± 0.3847 |
| Transformer with interaction | after | 10.6003 ± 0.6481 | 0.6178 ± 0.0138 | 0.3462 ± 0.0072 | 75.0971 ± 0.3847 |
| Gated Transformer | before | 11.3529 ± 0.1673 | 0.6607 ± 0.0283 | 0.3666 ± 0.0176 | 73.3379 ± 2.2539 |
| Gated Transformer | after | 10.7903 ± 0.3784 | 0.6509 ± 0.0278 | 0.3641 ± 0.0169 | 73.3379 ± 2.2539 |

## Per-target calibration effects

Each cell gives before → after. Decision-layer values are means across three seeds.

| Model | Dataset | ECE (%) | NLL | Brier |
|---|---|---:|---:|---:|
| Original UniMC | sst2 | 1.767 → 1.500 | 0.1950 → 0.1951 | 0.1171 → 0.1174 |
| Original UniMC | cb | 17.576 → 18.509 | 0.7192 → 0.7200 | 0.4530 → 0.4548 |
| Original UniMC | copa | 6.782 → 5.781 | 0.1474 → 0.1388 | 0.0751 → 0.0714 |
| Original UniMC | hellaswag | 9.821 → 11.759 | 1.1167 → 1.1466 | 0.5807 → 0.5876 |
| Residual MLP | sst2 | 4.983 → 6.004 | 0.2127 → 0.2196 | 0.1191 → 0.1219 |
| Residual MLP | cb | 15.863 → 15.451 | 1.0588 → 1.0490 | 0.6598 → 0.6532 |
| Residual MLP | copa | 6.793 → 7.507 | 0.1497 → 0.1574 | 0.0773 → 0.0804 |
| Residual MLP | hellaswag | 14.538 → 12.828 | 1.1793 → 1.1469 | 0.6038 → 0.5961 |
| Transformer without interaction | sst2 | 4.317 → 5.641 | 0.1999 → 0.2096 | 0.1112 → 0.1148 |
| Transformer without interaction | cb | 23.507 → 20.278 | 0.9358 → 0.9363 | 0.5783 → 0.5771 |
| Transformer without interaction | copa | 5.503 → 6.784 | 0.1456 → 0.1565 | 0.0739 → 0.0781 |
| Transformer without interaction | hellaswag | 15.139 → 12.516 | 1.1910 → 1.1445 | 0.6089 → 0.5974 |
| Transformer with interaction | sst2 | 5.138 → 7.073 | 0.2140 → 0.2277 | 0.1189 → 0.1244 |
| Transformer with interaction | cb | 17.708 → 17.009 | 0.9648 → 0.9630 | 0.5952 → 0.5922 |
| Transformer with interaction | copa | 6.329 → 6.572 | 0.1421 → 0.1547 | 0.0730 → 0.0776 |
| Transformer with interaction | hellaswag | 14.969 → 11.747 | 1.1804 → 1.1257 | 0.6042 → 0.5905 |
| Gated Transformer | sst2 | 4.740 → 6.383 | 0.2109 → 0.2220 | 0.1170 → 0.1213 |
| Gated Transformer | cb | 18.938 → 16.607 | 1.0624 → 1.0534 | 0.6561 → 0.6495 |
| Gated Transformer | copa | 5.871 → 6.979 | 0.1498 → 0.1614 | 0.0768 → 0.0814 |
| Gated Transformer | hellaswag | 15.863 → 13.193 | 1.2195 → 1.1666 | 0.6164 → 0.6042 |

## Cross-task findings

Across-seed mean target-macro ECE, NLL and Brier decreased together for 4 of the four decision-layer architectures. Target-macro ECE changes (after minus before) were Residual MLP -0.096 percentage points, Transformer without interaction -0.812 percentage points, Transformer with interaction -0.436 percentage points, Gated Transformer -0.563 percentage points.

Original UniMC had T=0.917180. Its source calibration NLL fell from 0.590412 to 0.589126. Its target-macro ECE changed from 8.986% to 9.387%, NLL from 0.5446 to 0.5501, and Brier from 0.3065 to 0.3078. This source-fitted temperature sharpened the probability distribution. On HellaSwag, original UniMC already had mean confidence above its accuracy, and further sharpening increased its probability error.

Decision-layer temperatures ranged from 1.0175 to 1.1873, softening their distributions. HellaSwag had lower mean ECE, NLL and Brier for every decision-layer architecture. SST-2 and COPA had higher mean NLL and Brier after this source-fitted softening. These tasks had different confidence-error patterns, so transfer effects varied by task. The per-target table retains both improvements and degradations.

Post-calibration target-macro minima were ECE for Original UniMC, NLL for Original UniMC, BRIER for Original UniMC. The three-seed summaries describe training-seed variation. The task-level changes are descriptive estimates from the fixed evaluation sets.

## Fitted temperatures

| Model / seed | T | Source calibration NLL before | After |
|---|---:|---:|---:|
| unimc | 0.917180 | 0.590412 | 0.589126 |
| mlp/seed0 | 1.144004 | 0.584587 | 0.580870 |
| mlp/seed1 | 1.068556 | 0.573586 | 0.572719 |
| mlp/seed2 | 1.017503 | 0.576481 | 0.576423 |
| tf_noint/seed0 | 1.131084 | 0.579165 | 0.575903 |
| tf_noint/seed1 | 1.101591 | 0.575356 | 0.573430 |
| tf_noint/seed2 | 1.121949 | 0.581936 | 0.579240 |
| tf_int/seed0 | 1.131948 | 0.579914 | 0.576812 |
| tf_int/seed1 | 1.123697 | 0.578865 | 0.576088 |
| tf_int/seed2 | 1.187312 | 0.580419 | 0.573935 |
| tf_gate/seed0 | 1.090069 | 0.578330 | 0.576856 |
| tf_gate/seed1 | 1.123607 | 0.577580 | 0.574824 |
| tf_gate/seed2 | 1.158175 | 0.585593 | 0.581076 |

## Verification and interpretation

All 168 prediction files matched the final archive's SHA-256 hashes. Source calibration IDs matched data/splits.json exactly and were disjoint from the source model-selection IDs. Labels, option counts, and original UniMC logits agreed across model and seed copies. Saved predictions matched the option-logit argmax.

Every fitted temperature reduced pooled source calibration NLL. Positive-temperature argmax checks passed for every evaluation batch. Target probability metrics were calculated with frozen source-fitted temperatures.

The source calibration questions were held out from the group's decision-layer training and model selection. Official UniMC's upstream MC training included these source tasks. The experiment therefore measures transfer from source-task calibration to the four target tasks.

CB has 56 questions and COPA has 100. Their ECE estimates and reliability curves have limited sample support. Per-bin counts are provided in reliability_bins.csv. Reliability figures show seed 0 for each decision layer, while all seeds contribute to summary tables.

## Files and reproduction

- temperatures.json contains model/checkpoint identities, source input hashes and fitted temperatures.
- metrics.csv and metrics.json contain per-run, per-dataset results, including source calibration metrics.
- target_macro_by_seed.csv and target_macro_summary.csv contain the task-equal target summaries.
- per_dataset_summary.csv contains seed means and sample standard deviations for every dataset.
- reliability_bins.csv contains counts and calibration statistics for all confidence bins.
- figures/ contains five reliability diagrams and one target-metric summary.
- audit.json and run_metadata.json record data checks and implementation identities.

From the repository root, run `python -m calibration.run_calibration`. Use `--temperatures-file calibration/results/temperatures.json --output <fresh-directory>` to replay evaluation with the frozen temperature parameters.

## Reference

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML. [Paper](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).
