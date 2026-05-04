# Asteria Ablation Study Report

## Run Configuration
| Parameter | Value |
|---|---|
| Sample count | 100 |
| Sample seed | 7 |
| Max seed rows | 20 |
| Model | watsonx/meta-llama/llama-3-3-70b-instruct |
| Skip summary | True |

## Latency Summary
| Metric | Baseline (all OFF) | Optimized (all ON) |
|---|---|---|
| Runs | 80 | 80 |
| Avg | 68.68s | 33.06s |
| Median | 34.10s | 9.80s |
| Min | 6.73s | 0.26s |
| Max | 398.73s | 230.78s |

## Aggregated Improvement
| Metric | Value |
|---|---|
| Cache Hits | 36 / 80 (45.0%) |
| Mean Speedup (Hit Rows) | 59.32x |
| Geom-Mean Speedup | 33.25x |
| Mean Latency Reduction | 95.0% |
| Mean Miss Overhead | -15.78s |

## Reliability
| TP | FP | FN | TN | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|---|
| 27 | 9 | 21 | 23 | 0.7500 | 0.5625 | 0.6429 | 0.7188 |

## Per-Row Results
| ID | Tier | Baseline (s) | Optimized (s) | Hit |
|---|---|---|---|---|
| 643 | paraphrase | 25.21 | 83.60 | no |
| 644 | paraphrase | 35.26 | 29.08 | no |
| 645 | paraphrase | 34.13 | 80.47 | no |
| 646 | paraphrase | 182.55 | 56.36 | no |
| 647 | paraphrase | 63.20 | 1.06 | YES |
| 648 | paraphrase | 398.73 | 1.13 | YES |
| 649 | paraphrase | 33.45 | 1.92 | YES |
| 650 | paraphrase | 65.92 | 25.26 | no |
| 651 | paraphrase | 66.23 | 2.74 | YES |
| 652 | paraphrase | 22.79 | 0.63 | YES |
| 653 | paraphrase | 7.30 | 0.60 | YES |
| 654 | paraphrase | 10.52 | 0.39 | YES |
| 655 | paraphrase | 51.35 | 199.32 | no |
| 656 | paraphrase | 170.92 | 130.33 | no |
| 657 | paraphrase | 27.86 | 41.82 | no |
| 658 | paraphrase | 164.23 | 1.22 | YES |
| 659 | paraphrase | 168.07 | 1.15 | YES |
| 660 | paraphrase | 34.34 | 1.96 | YES |
| 661 | paraphrase | 150.73 | 2.56 | YES |
| 662 | paraphrase | 18.80 | 16.27 | no |
| 663 | paraphrase | 17.29 | 4.52 | YES |
| 664 | paraphrase | 8.95 | 0.35 | YES |
| 665 | paraphrase | 8.92 | 0.53 | YES |
| 666 | paraphrase | 8.45 | 0.93 | YES |
| 667 | paraphrase | 9.83 | 1.01 | YES |
| 668 | paraphrase | 44.17 | 1.46 | YES |
| 669 | paraphrase | 37.99 | 17.78 | no |
| 670 | paraphrase | 346.77 | 227.22 | no |
| 671 | paraphrase | 11.36 | 2.69 | YES |
| 672 | paraphrase | 16.01 | 11.74 | no |
| 673 | paraphrase | 48.44 | 0.64 | YES |
| 674 | paraphrase | 34.07 | 0.46 | YES |
| 675 | paraphrase | 10.29 | 0.65 | YES |
| 676 | paraphrase | 8.70 | 0.33 | YES |
| 677 | paraphrase | 105.54 | 51.48 | no |
| 678 | paraphrase | 122.51 | 230.78 | no |
| 679 | paraphrase | 18.73 | 17.83 | no |
| 680 | paraphrase | 56.43 | 33.93 | no |
| 681 | paraphrase | 28.14 | 22.71 | no |
| 682 | paraphrase | 132.64 | 2.94 | YES |
| 683 | paraphrase | 74.82 | 1.61 | YES |
| 684 | paraphrase | 180.75 | 89.79 | no |
| 685 | paraphrase | 72.46 | 28.12 | no |
| 686 | paraphrase | 34.54 | 121.36 | no |
| 687 | paraphrase | 19.99 | 1.27 | YES |
| 688 | paraphrase | 44.99 | 14.56 | no |
| 689 | paraphrase | 23.63 | 0.87 | YES |
| 690 | paraphrase | 17.72 | 0.39 | YES |
| 691 | paraphrase | 28.66 | 0.43 | YES |
| 692 | paraphrase | 21.84 | 18.84 | no |
| 693 | paraphrase | 14.68 | 0.34 | YES |
| 694 | paraphrase | 165.51 | 21.13 | no |
| 695 | paraphrase | 39.83 | 1.19 | YES |
| 696 | paraphrase | 29.29 | 33.19 | no |
| 697 | paraphrase | 24.19 | 28.31 | no |
| 698 | paraphrase | 146.43 | 0.45 | YES |
| 699 | paraphrase | 280.43 | 1.44 | YES |
| 700 | paraphrase | 174.25 | 76.77 | no |
| 701 | paraphrase | 37.60 | 39.49 | no |
| 702 | paraphrase | 54.86 | 21.10 | no |
| 703 | paraphrase | 150.35 | 30.74 | no |
| 704 | paraphrase | 10.10 | 8.85 | no |
| 705 | paraphrase | 14.92 | 14.05 | no |
| 706 | paraphrase | 53.40 | 43.43 | no |
| 707 | paraphrase | 12.29 | 10.74 | no |
| 708 | paraphrase | 7.09 | 3.49 | no |
| 709 | paraphrase | 23.37 | 1.91 | YES |
| 710 | paraphrase | 73.14 | 87.33 | no |
| 711 | paraphrase | 32.40 | 0.79 | YES |
| 712 | paraphrase | 30.00 | 27.02 | no |
| 713 | paraphrase | 6.73 | 4.93 | no |
| 714 | paraphrase | 9.59 | 7.26 | no |
| 715 | paraphrase | 74.26 | 166.47 | no |
| 716 | paraphrase | 29.80 | 34.92 | no |
| 717 | paraphrase | 25.65 | 84.01 | no |
| 718 | paraphrase | 22.50 | 2.19 | YES |
| 719 | paraphrase | 207.19 | 73.37 | no |
| 720 | paraphrase | 81.09 | 30.84 | no |
| 721 | paraphrase | 13.00 | 0.26 | YES |
| 722 | paraphrase | 324.00 | 203.78 | no |