# Combined-HIF-Detector (中英文说明 / Bilingual README)

## 项目简介 / Overview
本项目用于电力配电网高阻接地故障（HIF）的**定位**研究。代码实现了多种方法并支持训练、评估和对比。
This project focuses on **localizing** high-impedance faults (HIF) in power distribution grids. It implements multiple methods and supports training, evaluation, and comparison.

## 能做什么 / What It Can Do
- 从 `data/` 中的仿真轨迹数据构建训练/验证集。
- 训练 LSTM 分类器并输出模型与日志。
- 评估 LSTM、动态系统、Koopman 方法的 top‑k 精度与窗口长度趋势。
- 生成对比图与诊断图（误差分布、残差 Koopman 热图等）。
- 支持多种实验配置与可复现实验脚本。

- Build train/validation splits from simulated trajectories in `data/`.
- Train LSTM classifiers and save checkpoints/logs.
- Evaluate LSTM, dynamic-system, and Koopman methods with top‑k accuracy and window-length trends.
- Generate comparison and diagnostic figures (error distributions, residual Koopman heatmaps, etc.).
- Support multiple experiment configs and reproducible scripts.

## 目录结构 / Directory Map
- `configs/`: 按方法组织的配置文件。
- `data/`: 输入数据（`.npy` 轨迹文件）。
- `scripts/`: 训练/评估脚本。
- `src/common/`: 共享预处理与公共工具。
- `src/lstm/`: LSTM 模型、训练与评估。
- `src/dynamic/`: 动态系统方法。
- `src/koopman/`: Koopman 与 residual Koopman 方法。
- `src/experimental/`: 实验性模块（如 Seq2Seq）。
- `src/eval_runner.py`: 统一评估入口。
- `test/`: 分析与诊断脚本。
- `tools/`: 结果可视化与对比工具。
- `notes/`: 实验报告与图表。

- `configs/`: Method-organized config files.
- `data/`: Input data (`.npy` trajectories).
- `scripts/`: Train/eval scripts.
- `src/common/`: Shared preprocessing and utilities.
- `src/lstm/`: LSTM model, train, and eval modules.
- `src/dynamic/`: Dynamic-system method.
- `src/koopman/`: Koopman and residual Koopman methods.
- `src/experimental/`: Experimental modules (for example Seq2Seq).
- `src/eval_runner.py`: Unified evaluation entrypoint.
- `test/`: Analysis and diagnostic scripts.
- `tools/`: Visualization and comparison utilities.
- `notes/`: Reports and figures.

## 快速开始 / Quick Start
Canonical usage is summarized in [TRAIN_EVAL_GUIDE.md](./TRAIN_EVAL_GUIDE.md).

### 1. LSTM 训练与评估 / LSTM Train & Eval
训练 / Train:
```bash
bash scripts/train/lstm.sh
```
评估 / Eval:
```bash
bash scripts/eval/lstm.sh
```
或使用统一评估 / Or unified eval:
```bash
python src/eval_runner.py \
  --method lstm \
  --config configs/lstm/classifier.json \
  --model_path checkpoints/lstm_classifier/2000/best_model.pth \
  --save_csv evaluations/lstm_eval.csv
```

### 2. 动态系统方法（已知输入）/ Dynamic System (Known Inputs)
```bash
bash scripts/eval/dynamic_known_control.sh
```
或统一评估 / Or unified eval:
```bash
python src/eval_runner.py \
  --method dynamic \
  --config configs/lstm/classifier.json \
  --save_csv evaluations/dynamic_eval.csv
```

### 3. Koopman 方法 / Koopman Methods
训练 / Train:
```bash
bash scripts/train/koopman_phi.sh
```
残差 Koopman 训练 / Residual Koopman training:
```bash
bash scripts/train/koopman_residual.sh
```
评估 / Eval:
```bash
bash scripts/eval/koopman_phi.sh
```
残差 Koopman 评估 / Residual Koopman eval:
```bash
bash scripts/eval/koopman_residual.sh
```

### 4. Seq2Seq 训练（用于未知输入场景）/ Seq2Seq Training (Unknown Inputs)
```bash
bash scripts/experimental/seq2seq_simple.sh
```

## 数据与预处理要点 / Data & Preprocessing Notes
- 数据位于 `data/`，格式为 `.npy`，包含 `signals` 和 `ErrorType`。
- 状态与输入通道在 `src/common/dataloader.py` 内拆分：
  - 状态 `x = signals[:, :-6]`
  - 控制 `u = signals[:, -6:-4]`
- 默认流程包括：裁剪前 1000 步、采样、滑窗、标准化、PCA(2D)。

- Data lives in `data/` as `.npy` files containing `signals` and `ErrorType`.
- Channel split is defined in `src/common/dataloader.py`:
  - State `x = signals[:, :-6]`
  - Control `u = signals[:, -6:-4]`
- Default pipeline: trim first 1000 steps, downsample, windowing, standardization, PCA(2D).

## 结果说明（摘要）/ Results (Summary)
更完整结果请见 `notes/` 下的 LaTeX 报告：
- `notes/main.tex`：初步结果与已知/未知输入对比。
- `notes/lstm_vs_dynamic.tex`：LSTM 与动态方法、Koopman 的对比分析。

For full results, see LaTeX reports in `notes/`:
- `notes/main.tex`: preliminary results with known/unknown inputs.
- `notes/lstm_vs_dynamic.tex`: LSTM vs dynamic/Koopman comparisons.

## 常见问题 / Common Issues
- 部分旧脚本引用的评估文件不存在，见 `TRAIN_EVAL_GUIDE.md` 中的 stale scripts 说明。
- 旧的 stale scripts 已从仓库移除。
- 当前脚本目录已经按 `scripts/train`、`scripts/eval`、`scripts/experimental` 重组。
- 脚本中的 checkpoint 路径可能需要按实际训练输出调整。

- Use `src/eval_runner.py` as the main unified evaluation entry.
- `eval_runner.py` default Koopman config path does not exist in this repo; pass one explicitly.
- Checkpoint paths in scripts may need to be updated to match your outputs.

## 依赖 / Dependencies
- `python` 3.9+
- `torch`, `numpy`, `scikit-learn`, `pandas`
- `matplotlib`, `seaborn`, `tqdm`
- `accelerate`, `wandb` (optional)

---
如需更详细的实验复现步骤或参数说明，可参考 `notes/project_handover.md`。
For more detailed reproduction steps and parameters, see `notes/project_handover.md`.
