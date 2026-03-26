## 🚀 Evaluation Guide

Follow these steps to evaluate our model on various benchmarks.

### 1. Prepare Model & Data
- **Download the Model**: Obtain our pre-trained model weights.
- **Download Datasets**: Download the required evaluation datasets and update the corresponding paths in your configuration.
  - *Note*: We have already provided the annotations for OVOBench, StreamingBench, and VideoHolmes. However, you still need to download the original video folders and update the video paths in `lmms_eval\tasks\[TASK]\utils.py`. For other datasets, please follow the standard Hugging Face format.

### 2. Environment Setup
Navigate to the evaluation directory and install the required package in editable mode:

```bash
cd eval/lmms-eval/
pip install -e .
```
### 3. Configure & Run
Run the script below to test all benchmarks sequentially. The results will be written to the `/eval/result` directory. Please check the .log files for the evaluation results.

You can comment out the benchmarks you do not want to run in `scripts/run.sh` to simplify the testing process.
```bash
cd ../
bash scripts/run.sh
```