# DiffusionGemma-26B-A4B RQ1 运行手册（the HPC cluster）

目标：把 `google/diffusiongemma-26B-A4B-it` 在 **RQ1（unified RefineID benchmark）** 上跑通，
产出非空预测 → LLM-as-Judge → 重建 Table II / 热力图 / CIS，并把图塞进论文。

约定：`[the HPC cluster]` 在集群 login/compute 节点跑；`[Mac]` 在本地（出图脚本是 Mac 绝对路径）。

> 关键事实
> - 模型 bf16 权重 ~52GB，**必须 96GB RTX PRO 6000**（A40/L40 48GB 必 OOM；SLURM 有 VRAM 闸门）。
> - DiffusionGemma 需要 `diffusion_gemma` 架构的 transformers（钉 5.11.0），装在**独立**
>   `$UMBRELLA/pylibs_dgemma` 树并 PREPEND 到 `PYTHONPATH`；**别动**主环境钉死的 4.57.1。
> - `.generate()` 就是它的 diffusion sampler，**不要**用 `diffusion_generate`。
> - 模型公开（Apache-2.0），下模型本身不需要 HF_TOKEN；但 **Gemma-2-27B judge 是 gated**，
>   跑 5-judge 时需要 `HF_TOKEN`。

---

## TL;DR — 一键跑完（无人值守）

```bash
ssh <cluster>
cd /path/to/CoReFusion && git pull && mkdir -p logs
PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh          # 一次性，装库+拉~52GB权重
HF_TOKEN=hf_xxx bash server/jobs/dgemma_rq1_all.sh        # 冒烟→全量→judge×5→重建，全自动
```
`dgemma_rq1_all.sh` 用 SLURM `afterok` 串四步，自动探测 96GB 卡。**冒烟是门控**：若预测仍
全空，冒烟 job 以非零退出，后续全量/judge 自动不跑（不浪费 24h）——这时去看
`logs/DiffusionGemma-smoke-*.out` 里的 `raw[0]`，按第 1 步的表判断是生成还是抽取问题。
只想要预测不跑 judge：`SKIP_LJ=1 bash server/jobs/dgemma_rq1_all.sh`（无需 HF_TOKEN）。

全链结束（`squeue` 清空）后：
```bash
source server/env_cluster.sh && python experiments/run_llm_judge_unified.py --combine-only
```
然后把 3 个 CSV 拉回 Mac 出图（见第 5–6 步）。

下面是**分步版**（想手动控制/调试时用）。

---

## 0. 前置（一次性）

```bash
# [Mac] 先把改动推上去（代码改动走 git；results/* 是 gitignore 的）
cd ~/Desktop/CoRefusion
git add experiments/benchmark_diffusiongemma.py analysis/build_fusion_dataset.py \
        analysis/viz_leaderboard.py analysis/viz_heatmap.py \
        server/jobs/DiffusionGemma-smoke.slurm docs/diffusiongemma_*.md
git commit -m "fix: DiffusionGemma RQ1 generate/extract + instrument; analysis auto-include"
git push

# [the HPC cluster] 拉代码
ssh <cluster>
cd /path/to/CoReFusion
git pull
mkdir -p logs

# [the HPC cluster] 一次性装 pylibs_dgemma（含 ~52GB 预下载）
PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh
# 末尾应打印 "OK: diffusion_gemma architecture available"

# [the HPC cluster] 找 96GB 卡的 GRES 型名（填进后面 --gres）
sinfo -o "%N %G" | grep -iE "6000|pro"
```

---

## 1. 冒烟 + 诊断（先确认非空，再跑全量）

```bash
# [the HPC cluster] 把 <型名> 换成上一步查到的（如 nvidia_rtx_pro_6000）
sbatch --gres=gpu:<型名>:1 server/jobs/DiffusionGemma-smoke.slurm
# 跟踪：
tail -f logs/DiffusionGemma-smoke-*.out
```

看 `dgemma debug` 段落（也写在 `results/diffusiongemma_smoke/debug.jsonl`），按下表判断：

| 现象 | 含义 | 处理 |
|------|------|------|
| `raw` 非空 **且** `pred` 非空 | **跑通了** | 直接进第 2 步全量 |
| `raw` 非空、`pred` 空 | 抽取问题（channel/标记与正则不符） | 把真实 `raw` 贴给我，按真实 token 改 `_extract`/正则 |
| `raw` 空、`new_len=0` | 生成问题（prompt/模板/canvas） | 看 `prompt_tail` 是否正常渲染；试 `--max-new-tokens 512`；确认 thinking 是否被禁 |
| `out_type` 不是 tensor 且 `seq_shape` 怪 | 返回类型问题 | 代码已对 `out.sequences` / tensor 双兼容，把 `out_type`/`seq_shape` 贴我 |

> 这一步就是当初一直没做的"抓 `raw[0]`"。一把日志就能定位到底是**生成**还是**抽取**坏了。

---

## 2. 全量 RQ1 推理（n=1000）

```bash
# [the HPC cluster] 写 results/unified_refineID/predictions/DiffusionGemma-26B-A4B.csv
sbatch --gres=gpu:<型名>:1 server/jobs/DiffusionGemma-26B-A4B.slurm
tail -f logs/DiffusionGemma-26B-A4B-*.out
```

注意：per-site 迭代 × 最多 ~61 个 mask × 1000 条，26B MoE 偏慢。job 给了 24h walltime；
如果不够，用 `--max-samples` 分段或先确认每条耗时。

---

## 3. LLM-as-Judge（5 judges，**主环境 4.57.1**，不要 PREPEND pylibs_dgemma）

```bash
# [the HPC cluster] judges 是 Qwen/Mistral/Gemma，跑在主环境；HF_TOKEN 用于 gated 的 Gemma-2-27B
export HF_TOKEN=...    # 你的 HF token
python experiments/run_llm_judge_unified.py --only DiffusionGemma-26B-A4B \
  --judge-model Qwen2.5-7B-Instruct  --judge-model Qwen2.5-14B-Instruct \
  --judge-model Qwen2.5-32B-Instruct --judge-model Mistral-Small-24B \
  --judge-model Gemma-2-27B-It
python experiments/run_llm_judge_unified.py --only DiffusionGemma-26B-A4B --combine-only
```
（GPU judge 也可以走 `server/jobs/` 的 SLURM；32B judge 需要 96GB 卡。）

---

## 4. 重建指标 + leaderboard（the HPC cluster，主环境，CPU 即可）

```bash
# [the HPC cluster] 不重新推理，只重算 metrics + leaderboard.csv
python experiments/run_all_refineID_unified.py --skip-inference
```

---

## 5. 把结果拉回 Mac

`results/*` 是 gitignore 的。两种方式：
- 我在 VPN+SSH 期间直接 `scp`/`rsync` 这几个小 CSV 回来（推荐，文件很小）：
  `results/unified_refineID/predictions/DiffusionGemma-26B-A4B.csv`、
  `results/unified_refineID/llm_judge/DiffusionGemma-26B-A4B__judge_*.csv`、
  `results/unified_refineID/leaderboard.csv`。
- 或在 the HPC cluster 把它们 copy 进 `transfer/`（tracked）→ commit/push → `[Mac] git pull`。

---

## 6. 出图 + CIS（Mac；出图脚本是 Mac 绝对路径）

```bash
# [Mac]
cd ~/Desktop/CoRefusion
python analysis/make_leaderboard.py          # -> figures/new/leaderboard_full.csv (+.md)
python analysis/build_fusion_dataset.py      # FAMILY 已含 DiffusionGemma -> fusion_consensus.csv
python analysis/metric_eval_final.py         # -> CIS / leaderboard_cis.csv / metric_eval F1-F4
python analysis/viz_leaderboard.py           # -> figures/new/fig6_leaderboard_bars.png
python analysis/viz_heatmap.py               # -> figures/new/fig7_metric_heatmap.png
```
viz 脚本已改成**数据驱动**：一旦预测非空，DiffusionGemma 自动出现（不再硬编码排除）；
全零行（仍空）才会被跳过。

---

## 7. 进论文（`~/Desktop/CoReFusion_ICSE27/conference_101719.tex`）

- `sec:models`：在 dLLM 名单里加 DiffusionGemma-26B-A4B（block-AR masked diffusion MoE，
  26B total / 4B active），模型计数 21→22、3 dLLM→4 dLLM。
- **Table II**（`tab:models`，~L411-413）：dLLM 区加一行，Type=`dLLM-BlockAR`，填 EM/Cons/CIS。
- **fig7 热力图**：把 Mac 上重生成的 `figures/new/fig7_metric_heatmap.png` 拷到论文 `figures/`。
- 脚注：prompted per-site naming（非 FIM/infill），与 base-model FIM 行不直接可比。
- RQ2/RQ3：按 `docs/diffusiongemma_rq_scope.md` 写一句 threats 说明为何不纳入。

---

## 常见坑速查

- `diffusion_gemma` import 失败 → `PYTHONPATH` 没吃到 pylibs_dgemma，或 setup 失败留空目录
  （job 会 fail-fast 提示）。重跑 `bash server/setup_dgemma_pylibs.sh`。
- CUDA OOM / 落在 48GB 卡 → 没带对 `--gres`；VRAM 闸门会在加载前报错并提示查型名。
- `AutoProcessor unavailable ... -> AutoTokenizer fallback` 是**正常**的（the HPC cluster 无 torchvision）；
  代码已对纯文本 fallback 用 string-content 渲染 chat 模板。
- 全量很慢 → 先 `--max-samples 50` 估一条耗时再决定是否分段。
