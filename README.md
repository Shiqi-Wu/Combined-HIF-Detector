# Combined-HIF-Detector

This repository contains the current Python implementation for high-impedance fault (HIF) localization in power distribution grids. In the present dataset, only a small set of discrete fault positions is available, so the first-stage formulation appears as a six-class fault-type classification problem. Conceptually, however, the intended task is fault localization.

## Scope

The repository supports three main method families:

- BiLSTM sequence classification
- Dynamic-system classification with known controls
- Koopman-based models, including a residual Koopman variant

It also includes one experimental Seq2Seq branch for unknown-input settings, but that branch is not part of the stable main workflow.

## Repository Layout

- `configs/`: Method-specific configuration files
- `configs/lstm/`: LSTM configs
- `configs/dynamic/`: Dynamic-system configs
- `configs/koopman/`: Koopman configs
- `configs/runtime/`: Runtime configs such as `accelerate`
- `data/`: Input `.npy` trajectory files
- `scripts/train/`: Stable training entry scripts
- `scripts/eval/`: Stable evaluation entry scripts
- `scripts/experimental/`: Experimental scripts
- `src/common/`: Shared data loading, preprocessing, and utility code
- `src/lstm/`: LSTM model, training, and evaluation
- `src/dynamic/`: Dynamic-system method
- `src/koopman/`: Koopman and residual Koopman methods
- `src/experimental/`: Experimental modules
- `src/eval_runner.py`: Unified evaluation entry point
- `notes/`: Handover notes and report material
- `tools/`: Plotting and comparison utilities
- `test/`: Analysis scripts and saved diagnostic outputs

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PyYAML
- accelerate
- wandb

## Data Format and Preprocessing

Data is expected under `data/` as `.npy` files. Each file stores a Python dictionary that includes at least:

- `signals`
- `ErrorType`

The shared loader in `src/common/dataloader.py` currently interprets the channels as:

- state channels: `signals[1000:, :-6]`
- control channels: `signals[1000:, -6:-4]`

The default preprocessing flow is:

1. Drop the first 1000 time steps.
2. Remove columns `[9, 21, 25, 39, 63]` from the state channels.
3. Downsample by `sample_step`.
4. Split each trajectory into non-overlapping windows.
5. Apply the scaling pipeline `x -> standardize -> PCA -> standardize`.
6. Standardize the control channels separately.

For the default configs, the label space has six classes. In the current data release, these classes are used as a proxy for discrete fault locations.

## Stable Workflows

Run commands from the repository root:

```bash
cd /Users/shiqi/Documents/PhD/Code/Project3-power-grid/Combined-HIF-Detector
```

### LSTM

Train:

```bash
bash scripts/train/lstm.sh
```

Evaluate:

```bash
bash scripts/eval/lstm.sh
```

Unified evaluation:

```bash
python src/eval_runner.py \
  --method lstm \
  --config_dir configs \
  --config lstm/classifier.json \
  --model_path checkpoints/lstm_classifier/2000/best_model.pth \
  --save_csv evaluations/lstm_eval.csv
```

### Dynamic-System Method

This method is evaluated in the known-control setting. It does not rely on a separately saved train checkpoint in the same way as the neural models; the system matrices are estimated during evaluation from the training split.

Evaluate:

```bash
bash scripts/eval/dynamic_known_control.sh
```

Unified evaluation:

```bash
python src/eval_runner.py \
  --method dynamic \
  --config_dir configs \
  --config lstm/classifier.json \
  --save_csv evaluations/dynamic_eval.csv
```

### Koopman

Train:

```bash
bash scripts/train/koopman_phi.sh
```

Evaluate:

```bash
bash scripts/eval/koopman_phi.sh
```

Unified evaluation:

```bash
python src/eval_runner.py \
  --method koopman \
  --config_dir configs \
  --config lstm/classifier.json \
  --koopman_config koopman/phi.json \
  --koopman_checkpoint checkpoints/koopman_phi/best_model.pt \
  --save_csv evaluations/koopman_phi_eval.csv
```

### Residual Koopman

Train:

```bash
bash scripts/train/koopman_residual.sh
```

Evaluate:

```bash
bash scripts/eval/koopman_residual.sh
```

## Experimental Workflow

The Seq2Seq branch remains experimental:

```bash
bash scripts/experimental/seq2seq_simple.sh
```

Do not treat it as a stable baseline for the main comparison.

## Evaluation Outputs

`src/eval_runner.py` is the main unified evaluation entry point for the stable methods. It supports:

- overall metrics
- top-k accuracy
- window-length trend analysis
- separate saved CSV outputs

The CLI now supports `--config_dir`, so configs can be passed as paths relative to `configs/`.

## Important Notes

- The core research goal is fault localization. The current six-class setup should be interpreted as a first-stage discrete-location proxy, not as the final problem definition.
- `scripts/train/`, `scripts/eval/`, and `scripts/experimental/` are the maintained shell entry points.
- `src/experimental/seq2seq_train.py` remains incomplete as a production-ready pipeline and should be treated as exploratory code.
- Paths in the provided stable scripts are already aligned with the current repository structure, but checkpoint files must still exist at the expected locations.

## Additional Documentation

For more detail, see:

- `TRAIN_EVAL_GUIDE.md`
- `notes/project_handover.md`
- the LaTeX handover material in `notes/`
