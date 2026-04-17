# FluxMem StreamingBench Results

This note records the completed FluxMem StreamingBench run found under `/home/sjs/FluxMem`.

## Result Files

- FluxMem: `/home/sjs/FluxMem/eval_results/streamingbench_20260406_075741/cc594898137f460bfe9f0759e9844b3ce807cfb5_scores.json`
- Baseline: `/home/sjs/FluxMem/eval_results/streamingbench_baseline_20260408_030946/cc594898137f460bfe9f0759e9844b3ce807cfb5_scores.json`

## Overall

| Method | Total | Correct | Accuracy |
|---|---:|---:|---:|
| Baseline (`use_fluxmem=false`) | 2499 | 1858 | 74.35 |
| FluxMem (`use_fluxmem=true`) | 2499 | 1906 | 76.27 |

FluxMem improves over the baseline by **+1.92 accuracy points** on StreamingBench.

## Per-Category Comparison

| Category | Baseline Acc. | FluxMem Acc. | Delta |
|---|---:|---:|---:|
| Object Perception | 77.51 | 80.22 | +2.71 |
| Causal Reasoning | 80.31 | 80.31 | +0.00 |
| Text-Rich Understanding | 81.00 | 84.11 | +3.12 |
| Attribute Perception | 82.69 | 85.26 | +2.56 |
| Event Understanding | 74.21 | 77.36 | +3.14 |
| Action Perception | 65.91 | 69.32 | +3.41 |
| Spatial Understanding | 64.23 | 65.85 | +1.63 |
| Clips Summarize | 82.02 | 81.39 | -0.63 |
| Prospective Reasoning | 77.78 | 80.56 | +2.78 |
| Counting | 53.19 | 52.13 | -1.06 |

## Run Configuration Notes

FluxMem run summaries report:

```text
use_fluxmem: true
frame_sampling: uniform
short_frames: 8
medium_frames: 64
```

Baseline run summaries report:

```text
use_fluxmem: false
frame_sampling: uniform
short_frames: 8
medium_frames: 16
```

## Interpretation

FluxMem provides a modest but consistent improvement over the baseline on most categories. The largest gains are in action perception, event understanding, text-rich understanding, and prospective reasoning. It slightly underperforms the baseline on clips summarization and counting.

For the new budgeted write-time memory project, this result is a useful fixed-memory / streaming-memory reference point. A strong memory-call method should be compared against both the baseline and this FluxMem run under matched or explicitly reported memory budgets.
