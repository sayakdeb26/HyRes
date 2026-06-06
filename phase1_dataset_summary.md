# Phase-1 Dataset Summary Report

This report provides a summary of the curated Phase-1 dataset for the gesture recognition pipeline.

## Target Gestures & Label Mapping
| Original Gesture Name | Mapped Name | Label |
| --- | --- | --- |
| Swiping Left | Swipe Left | 0 |
| Swiping Right | Swipe Right | 1 |
| Rolling Hand Forward | Rolling Hand Forward | 2 |
| Stop Sign | Stop Sign | 3 |

## Split Distributions
| Split | Swipe Left | Swipe Right | Rolling Hand Forward | Stop Sign | Total |
| --- | --- | --- | --- | --- | --- |
| train_70 | 2913 | 2859 | 2892 | 3036 | 11700 |
| inc10_a | 416 | 408 | 413 | 434 | 1671 |
| inc10_b | 416 | 409 | 413 | 434 | 1672 |
| inc10_c | 417 | 408 | 414 | 433 | 1672 |
| validation | 494 | 486 | 521 | 536 | 2037 |
| **Total** | **4656** | **4570** | **4653** | **4873** | **18752** |

## Integrity Verification Status
- **Overlap Check**: PASSED (No overlapping video IDs across splits)
- **Link Verification Check**: PASSED (All symbolic links resolved successfully)
- **Overall Status**: PASS

## File Artifacts Generated
- **Dataset Manifest**: [dataset_manifest_phase1.csv](dataset_manifest_phase1.csv)
- **Split Class Distribution Report**: [split_report.csv](split_report.csv)
- **Detailed Integrity Audit**: [dataset_integrity_report.txt](dataset_integrity_report.txt)

---  
*Note: Video folders under `DataSet_Full/phase1` are symlinked to conserve disk space.*  
