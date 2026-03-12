# Train and Eval Guide

This repository currently has three stable method families for HIF localization:

1. `BiLSTM`
2. `Dynamic system (known-control)`
3. `Koopman`

`Koopman` has two concrete variants in code:
- standard parametric Koopman
- residual Koopman

The safest evaluation entrypoint is `src/eval_runner.py` for `lstm`, `dynamic`, and standard `koopman`.
Residual Koopman uses its own evaluator: `src/koopman/residual_eval.py`.

## 1. Common data pipeline

All methods use the same preprocessing from `src/common/dataloader.py`:

- load `.npy` trajectories from `data/`
- discard the first 1000 samples
- define state as `signals[:, :-6]`
- define known HIF input as `signals[:, -6:-4]`
- drop columns `[9, 21, 25, 39, 63]`
- downsample by `sample_step`
- split train/val with stratification
- cut trajectories into non-overlapping windows
- apply `scale -> PCA(2D) -> scale` on the state

## 2. Method A: BiLSTM

### Train

Recommended script:

```bash
bash scripts/train/lstm.sh
```

Direct command:

```bash
python src/lstm/train.py \
  --config configs/lstm/classifier.json
```

Default checkpoint:

```bash
checkpoints/lstm_classifier/2000/best_model.pth
```

### Eval

Recommended command:

```bash
python src/eval_runner.py \
  --method lstm \
  --config configs/lstm/classifier.json \
  --model_path checkpoints/lstm_classifier/2000/best_model.pth \
  --save_csv evaluations/lstm_eval.csv
```

The script below is also usable:

```bash
bash scripts/eval/lstm.sh
```

## 3. Method B: Dynamic system classifier

This is the current stable physics-informed baseline. It assumes the HIF-related input `u` is known from the dataset.

### Train

There is no separate training checkpoint for this method.
`K` and `B(p)` are estimated by least squares inside evaluation from the training split.

### Eval

Recommended script:

```bash
bash scripts/eval/dynamic_known_control.sh
```

Direct command:

```bash
python src/eval_runner.py \
  --method dynamic \
  --config configs/lstm/classifier.json \
  --save_csv evaluations/dynamic_eval.csv
```

Important:

- this is a `known-control` evaluation
- it uses the ground-truth input channels extracted from the dataset
- it is not the same as a full unknown-input deployment setting

## 4. Method C: Standard Koopman

### Train

Recommended script:

```bash
bash scripts/train/koopman_phi.sh
```

Direct command:

```bash
python src/koopman/phi_train.py \
  --config configs/koopman/phi.json
```

Default checkpoint:

```bash
checkpoints/koopman_phi/best_model.pt
```

### Eval

Recommended script:

```bash
bash scripts/eval/koopman_phi.sh
```

Direct command:

```bash
python src/eval_runner.py \
  --method koopman \
  --config configs/lstm/classifier.json \
  --koopman_config configs/koopman/phi.json \
  --koopman_checkpoint checkpoints/koopman_phi/best_model.pt \
  --save_csv evaluations/koopman_phi_eval.csv
```

## 5. Method C2: Residual Koopman

This is still part of the Koopman family, but it uses a separate trainer and evaluator.

### Train

Recommended script:

```bash
bash scripts/train/koopman_residual.sh
```

Direct command:

```bash
python src/koopman/residual_train.py \
  --config configs/koopman/residual.json
```

Default checkpoint:

```bash
checkpoints/koopman_residual/best_model.pt
```

### Eval

Recommended script:

```bash
bash scripts/eval/koopman_residual.sh
```

Direct command:

```bash
python src/koopman/residual_eval.py \
  --config configs/koopman/residual.json \
  --checkpoint checkpoints/koopman_residual/best_model.pt \
  --output_prefix evaluations/koopman_residual
```

## 6. Which scripts are stale

These scripts were stale and have been removed from the repository:

- `scripts/run_dynamic_system_eval.sh`
- `scripts/run_seq2seq_evaluation.sh`
- `scripts/run_state_based_evaluation.sh`

Reason:

- they point to evaluator files that do not exist in this repository
- they belong to older experimental paths that were not kept in sync

Use the commands above instead. The current script layout is:

- `scripts/train/`
- `scripts/eval/`
- `scripts/experimental/`

## 7. Recommended workflow

If you want the cleanest current workflow, use:

1. `BiLSTM` as the main trainable baseline
2. `Dynamic system (known-control)` as the physics-informed baseline
3. `Standard Koopman` and `Residual Koopman` as exploratory interpretable baselines

For side-by-side comparison:

```bash
python src/eval_runner.py \
  --method all \
  --config configs/lstm/classifier.json \
  --model_path checkpoints/lstm_classifier/2000/best_model.pth \
  --koopman_config configs/koopman/phi.json \
  --koopman_checkpoint checkpoints/koopman_phi/best_model.pt \
  --save_csv evaluations/all_methods.csv
```
