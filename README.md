# lLm — Looped VLM Phase 0

**目标**：验证 H1 —— *给 recurrent-depth LLM (Huginn/Ouro) 接上视觉后，loop 深度在 MMMU/MathVista/ScienceQA 类推理任务上是否带来可测增益*。
结果无论正反都决定下一步：继续 Phase 1 完整 Option 4，还是切换到 Phase 2 fallback paper。

---

## 为什么存在这个 repo

调研后发现，目前 recurrent-depth LM 领域已被以下工作占满：
Huginn / Ouro / MoR / Two-Scale Dynamics / LoopFormer (ICLR 2026) / 2604.11791 跨模型分析 / LoopViT / RD-VLA。
仍留下一个窄 gap：**looped VLM + 基于输入特征的 pre-inference depth 预测**。
但这个方向是否值得做，取决于更基础的假设：**loop 深度在视觉推理任务上有没有帮助？** 从未有人验证。
Phase 0 就是来回答这个问题的。

## 架构

```
LoopedVLM
├── CLIP ViT-L/14-336         (frozen)
├── MLP projector             (trainable, ~30M params)
└── Huginn-0125 / Ouro-1.4B   (frozen, num_steps 可控)
```

LLM 冻结是关键：否则 finetune 会破坏 pretrain 学到的 loop 动力学，H1 的验证就无意义了。

## 目录

```
lLm/
├── smoke_test.py                 # 0. 最先跑：验证 Huginn 的 inputs_embeds + num_steps 路径
├── configs/huginn_vlm.yaml       # 全部超参
├── src/
│   ├── model/looped_vlm.py       # CLIP + projector + frozen LLM wrapper
│   ├── data/llava_dataset.py     # LLaVA-1.5 pretrain data loader
│   ├── train/train_projector.py  # 只训 projector
│   └── eval/
│       ├── eval_mmmu.py
│       ├── eval_mathvista.py
│       └── eval_scienceqa.py
├── plot_results.py               # 画 accuracy vs num_steps 曲线 + 决策报告
├── scripts/
│   ├── setup_runpod.sh           # 环境 + 模型缓存预热
│   ├── download_llava.sh         # LLaVA-1.5 pretrain 数据
│   └── run_phase0.sh             # 端到端一键脚本
└── results/                      # eval JSON、曲线图、决策报告
```

## 在 RunPod A100 上跑

建议 A100 80GB 单卡（40GB 也能跑，需要把 batch 减半）。

```bash
# 1) clone
git clone https://github.com/ANTI-Tony/lLm.git
cd lLm

# 2) env + 模型预热
bash scripts/setup_runpod.sh

# 3) 数据（~14GB）
bash scripts/download_llava.sh

# 4) 一键 Phase 0
bash scripts/run_phase0.sh
```

或者分步：

```bash
# 先做最重要的一步 —— 烟雾测试
python smoke_test.py

# 训 projector（~12-16h on A100-80G，max_samples=200k）
python -m src.train.train_projector --config configs/huginn_vlm.yaml

# 评估（每个 benchmark ~1-2h，num_steps 扫 5 个值）
python -m src.eval.eval_mmmu      --config configs/huginn_vlm.yaml --projector checkpoints/projector_huginn/projector_final.pt
python -m src.eval.eval_mathvista --config configs/huginn_vlm.yaml --projector checkpoints/projector_huginn/projector_final.pt
python -m src.eval.eval_scienceqa --config configs/huginn_vlm.yaml --projector checkpoints/projector_huginn/projector_final.pt

# 画图 + 决策
python plot_results.py \
    --inputs results/mmmu.json results/mathvista.json results/scienceqa.json \
    --labels MMMU MathVista ScienceQA \
    --out results/phase0_curve.png
```

## Phase 0 判决规则（写在代码里）

`plot_results.py` 会读所有 eval JSON，对每个 benchmark 找出 peak 的 num_steps 和其相对于 step-1 baseline 的增益：

| 条件 | 结论 | 下一步 |
|------|------|--------|
| 任一 benchmark 有 peak @ num_steps > 1 且 Δacc ≥ 2pp | **H1 LIKELY HOLDS** | 进入 Phase 1（完整 Option 4） |
| 所有 benchmark 都平坦/单调 | **H1 APPEARS TO FAIL** | Phase 2 fallback："Why Latent Iteration Fails on Visual Reasoning" |

## 关键技术注意事项

- `transformers==4.54.1`（requirements 已固定），升到 4.56+ Ouro 会坏。
- Huginn `num_steps` 通过 `model(input_ids, num_steps=N)` 和 `model.generate(..., num_steps=N)` 传递，**不能放进 `GenerationConfig`**。
- Huginn 的 sweet spot 是 4-64 步；低于 4 答案会非常粗糙；本 repo sweep `[4, 8, 16, 32, 64]`。
- `smoke_test.py` 的 check-2（`inputs_embeds` + `num_steps` 同时使用）是 VLM 路径的技术前提。若这一步 FAIL，必须先 patch modeling_huginn 再继续。
- 训练时 `num_steps_for_training=16`，是性能/成本折中；后续 Phase 1 再做 ablation。

## 预算估算（A100-80GB 单卡，RunPod 约 $1.89/h）

三档预设，默认是 Fast。选择逻辑：Fast 就能做 H1 判决；只有在 Fast 结果模糊时才升级。

### 🏃 Fast Phase 0（默认 config，推荐）：**~5-6h, ~$12**
- projector: 50k 样本 · num_steps=8 → ~1.5h
- eval: 300+300+500 题 × [4,16,64] → ~3.5h
- 用途：决定 H1 成立与否

### 🚶 Standard Phase 0（`configs/huginn_vlm_standard.yaml`）：**~12-15h, ~$28**
- projector: 100k 样本 · num_steps=8 → ~3h
- eval: 900+1000+2000 题 × [4,8,16,32,64] → ~12h
- 用途：Fast 结果成立后，为论文拿到 publication-grade 曲线

### 🐢 Full（不推荐）：~45h, ~$85
- projector: 200k 样本 · num_steps=16 → ~35h
- 我原方案低估了，Huginn 在 num_steps=16 的 forward 等价 26B 算力，比 Vicuna-7B 的 LLaVA pretrain 贵 3-4 倍
- 只有在你想把 projector 做到 SOTA VLM 水平时才值得

## 下一步（Phase 0 完成后）

- H1 成立 → 把本 repo 作为 Phase 1 的底座，增加：input-feature 提取 / oracle depth labeling / predictor 训练 / 与 Ouro entropy gate 的 head-to-head。
- H1 不成立 → 改写论文方向为 negative-result empirical paper；复用本 repo 做 attention pattern / cross-loop CKA / representation drift 诊断。

---

详细背景、论文调研、假设链分析见 `~/Desktop/Looped_VLM_调研与分阶段方案.docx`。
