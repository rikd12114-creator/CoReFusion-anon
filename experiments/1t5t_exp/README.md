# 1t5t_exp — MASK Token Count Experiments

研究问题：将原始变量名替换成不同数量的 `<|mask|>` token，对扩散语言模型变量命名重构效果有何影响？

```
experiments/1t5t_exp/
├── README.md                          (本文件)
├── part1_token_length_analysis.py     # Part 1: GT变量名token长度分布分析
├── part2_static_token_ablation.py     # Part 2: 静态token数量消融实验 (1-5)
└── part3_dynamic_vs_static.py         # Part 3: 动态 vs 静态token数量对比

results/1t5t_exp/
├── part1_summary_*.csv                # Part 1 汇总
├── part1_dist_*.csv                   # Per-tokenizer 分布详情
├── part2_raw_*.csv                    # Part 2 原始预测结果
├── part2_summary_*.csv                # Part 2 汇总 (EM + LLM-Judge per k)
├── part3_raw_*.csv                    # Part 3 原始结果
├── part3_summary_*.csv                # Part 3 汇总 (动态 vs 静态)
└── figures/
    ├── part1_*_histogram.png/pdf
    ├── part1_*_cdf.png/pdf
    ├── part1_*_bar.png/pdf
    ├── part2_ablation_*.png/pdf
    ├── part3_dynamic_vs_static_*.png/pdf
    └── part3_k_distribution_*.png
```

---

## Part 1 — GT 变量名 Token 长度分析

**目的**：在 DiffuCoder 和 DreamCoder 各自的 tokenizer 下，统计 RefineID 数据集中 GT 变量名的 token 数量分布，为 Part 2 的消融范围提供依据。

### 运行命令

```bash
# 从项目根目录运行

# 完整数据集 (~25k 样本)
python experiments/1t5t_exp/part1_token_length_analysis.py \
    --data data/test.csv

# 快速验证 (前 1000 个样本)
python experiments/1t5t_exp/part1_token_length_analysis.py \
    --data data/test.csv \
    --max-samples 1000
```

### 输出
- `results/1t5t_exp/part1_summary_*.csv` — 均值、中位数、P90、各token长度占比
- `results/1t5t_exp/part1_dist_*.csv` — 详细分布表
- `results/1t5t_exp/figures/part1_*_histogram.png` — 直方图
- `results/1t5t_exp/figures/part1_*_cdf.png` — CDF 曲线
- `results/1t5t_exp/figures/part1_*_bar.png` — 分桶柱状图

---

## Part 2 — 静态 MASK Token 数量消融实验

**目的**：给定已知的 [MASK] 位置，测试将其替换为 k ∈ {1, 2, 3, 4, 5} 个 `<|mask|>` token 时，DiffuCoder 和 DreamCoder 的变量命名重构质量。

**评估指标**：
- **Exact Match (EM)**: 预测 == GT 字符串精确匹配
- **LLM-as-Judge (LJ)**: Qwen2.5-7B-Instruct 打分 (0/1)，语义可接受性

**固定参数**：diffusion steps = 32

### 运行命令

```bash
# ── 完整实验：两个模型，k=1~5，含 LLM Judge ────────────────────────────
python experiments/1t5t_exp/part2_static_token_ablation.py \
    --data data/test.csv \
    --models both \
    --mask-counts 1 2 3 4 5 \
    --steps 32 \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --max-samples 200

# ── 仅 Exact Match，跳过 LLM Judge（更快）────────────────────────────────
python experiments/1t5t_exp/part2_static_token_ablation.py \
    --data data/test.csv \
    --models both \
    --mask-counts 1 2 3 4 5 \
    --steps 32 \
    --no-judge \
    --max-samples 200

# ── 只测试 DiffuCoder ─────────────────────────────────────────────────────
python experiments/1t5t_exp/part2_static_token_ablation.py \
    --data data/test.csv \
    --models diffucoder \
    --mask-counts 1 2 3 4 5 \
    --steps 32 \
    --no-judge

# ── 只测试 DreamCoder ─────────────────────────────────────────────────────
python experiments/1t5t_exp/part2_static_token_ablation.py \
    --data data/test.csv \
    --models dreamcoder \
    --mask-counts 1 2 3 4 5 \
    --steps 32 \
    --no-judge

# ── 快速 smoke test（5个样本，k=1和3）───────────────────────────────────
python experiments/1t5t_exp/part2_static_token_ablation.py \
    --data data/test.csv \
    --mask-counts 1 3 \
    --steps 32 \
    --no-judge \
    --max-samples 5
```

### 输出
- `results/1t5t_exp/part2_raw_{model}_{k}tok_*.csv` — 每样本预测
- `results/1t5t_exp/part2_summary_*.csv` — 汇总表（每模型每k的EM和LJ）
- `results/1t5t_exp/figures/part2_ablation_*.png` — 消融对比柱状图

---

## Part 3 — 动态 vs 静态 Token 数量对比

**目的**：验证如果根据代码上下文动态决定 MASK token 数量，是否能比统一的静态 k 更好地恢复变量名。

**动态策略**:
1. **`dynamic_threshold`** — 基于实验C的熵阈值（F1-optimal τ）映射到 k：用 [MASK] 附近标识符的平均字符长度作为代理指标，近似模拟宽松/紧凑的阈值决策边界
2. **`dynamic_context`** — 用当前代码片段中**其他**变量名（非 [MASK] 位置）在 target tokenizer 下的平均 token 长度，四舍五入后裁剪到 [1,5]

**基准**：Part 2 最优的静态 k（可通过 `--static-k` 手动指定，或通过 `--part2-summary` 自动从 Part 2 结果选取）

### 运行命令

```bash
# ── 完整实验：两个模型，三种策略（static + 两种动态），含 LLM Judge ──────
python experiments/1t5t_exp/part3_dynamic_vs_static.py \
    --data data/test.csv \
    --models both \
    --static-k 3 \
    --steps 32 \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --max-samples 200

# ── 从 Part 2 的最佳 k 自动选取 static-k ─────────────────────────────────
python experiments/1t5t_exp/part3_dynamic_vs_static.py \
    --data data/test.csv \
    --models both \
    --part2-summary results/1t5t_exp/part2_summary_YYYYMMDD_HHMMSS.csv \
    --steps 32 \
    --no-judge \
    --max-samples 200

# ── 仅对比两种动态策略（不跑 static baseline）────────────────────────────
python experiments/1t5t_exp/part3_dynamic_vs_static.py \
    --data data/test.csv \
    --strategies dynamic_threshold dynamic_context \
    --steps 32 \
    --no-judge \
    --max-samples 200

# ── 快速 smoke test ───────────────────────────────────────────────────────
python experiments/1t5t_exp/part3_dynamic_vs_static.py \
    --data data/test.csv \
    --models diffucoder \
    --strategies static dynamic_context \
    --static-k 3 \
    --steps 32 \
    --no-judge \
    --max-samples 5
```

### 输出
- `results/1t5t_exp/part3_raw_{model}_{strategy}_*.csv` — 每样本结果（含 dynamic_k 列）
- `results/1t5t_exp/part3_summary_*.csv` — 汇总对比表
- `results/1t5t_exp/figures/part3_dynamic_vs_static_*.png/pdf` — 对比柱状图
- `results/1t5t_exp/figures/part3_k_distribution_*.png` — 动态 k 值分布图

---

## 推荐运行顺序

```bash
# Step 1: 先分析 GT token 长度（轻量，无需 GPU）
python experiments/1t5t_exp/part1_token_length_analysis.py --data data/test.csv

# Step 2: 消融静态 k（核心实验，需要 GPU）
python experiments/1t5t_exp/part2_static_token_ablation.py \
    --data data/test.csv --models both --mask-counts 1 2 3 4 5 \
    --steps 32 --no-judge --max-samples 200

# Step 3: 加 LLM Judge 重跑 Part 2（或单独评估）
python experiments/1t5t_exp/part2_static_token_ablation.py \
    --data data/test.csv --models both --mask-counts 1 2 3 4 5 \
    --steps 32 --judge-model Qwen/Qwen2.5-7B-Instruct --max-samples 200

# Step 4: 动态 vs 静态对比（利用 Part 2 最优 k）
python experiments/1t5t_exp/part3_dynamic_vs_static.py \
    --data data/test.csv --models both --steps 32 \
    --part2-summary results/1t5t_exp/part2_summary_<TIMESTAMP>.csv \
    --judge-model Qwen/Qwen2.5-7B-Instruct --max-samples 200
```

---

## 实验参数速查

| 参数 | 值 |
|---|---|
| Dataset | `data/test.csv` (RefineID, Java) |
| DiffuCoder | `apple/DiffuCoder-7B-Instruct` |
| DreamCoder | `Dream-org/Dream-Coder-v0-Instruct-7B` |
| Mask token | `<\|mask\|>` |
| Diffusion steps | **32** (固定) |
| k range (Part 2) | 1, 2, 3, 4, 5 |
| LLM Judge | `Qwen/Qwen2.5-7B-Instruct` |
| Metrics | Exact Match (EM), LLM-as-Judge (LJ) |
