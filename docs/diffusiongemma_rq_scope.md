# DiffusionGemma 的实验范围：为什么只进 RQ1，不进 RQ2 / RQ3

**结论先行**：把 `google/diffusiongemma-26B-A4B-it`（DiffusionGemma-26B-A4B）加入 **RQ1**
是合理且可比的；但它**不能**进入 RQ2 和 RQ3。原因**不是工作量**，而是**架构与实验语义不匹配**——
RQ2/RQ3 测量的是一种 DiffusionGemma 在架构上**做不到的操作**。下面逐条解释。

---

## 一句话区分三个 RQ 测的是什么

| RQ | 测量对象 | 依赖的模型能力 |
|----|----------|----------------|
| **RQ1** | **端到端任务准确率**：多处出现的同一标识符，重命名得对不对（EM / 一致性 / CIS / LLM-judge） | 只要能**给出每个 site 的预测名** |
| **RQ2** | **in-place 去混淆 / copy-bias 机制**：把多个不同标识符同时 mask 在原序列里，比较 "全 mask" vs "只 mask 目标" 两种上下文下的填充行为 | 必须能**在原序列任意位置就地 infill `<|mask|>`** |
| **RQ3** | **内部机制**：denoising 过程中，模型在标识符 token 位置上的**逐层 residual-stream 表征轨迹**（好名字 vs 坏名字的内部状态差异） | 必须存在**固定的、可探针的 in-place 标识符位置**，且有 denoising 轨迹可读 |

RQ1 只要"输出一个名字"，所以 DiffusionGemma 可以用 **prompted per-site**（一次问一个标识符）
的方式参与，论文里加一个脚注说明它是 prompted naming 而非 FIM/infill 即可。
RQ2/RQ3 测的是 **infill 这个操作本身**和**它在原位上的内部表征**——这正是 DiffusionGemma 没有的。

---

## DiffusionGemma 的架构事实（来自 `config.json` 与模型卡）

- `model_type = diffusion_gemma`，`architectures = [DiffusionGemmaForBlockDiffusion]`，
  `canvas_length = 256`，128-expert **MoE**，bf16 权重 ~52GB。
- 它是 **block-autoregressive（block diffusion）**：在 prompt **之后追加一块生成 canvas**，
  对**这块 canvas** 做 denoising；`.generate()` 本身就是它的 diffusion sampler。
- 它**没有** `model.diffusion_generate(...)`（那是 Dream/DiffuCoder 的 API）。
- 它**没有 FIM / base checkpoint**，只有 instruct 模型。
- 它**无法在输入序列的任意位置就地 infill `<|mask|>`**——这是它和 DiffuCoder/DreamCoder
  这类 **fixed-canvas masked dLLM** 的根本区别。

---

## 为什么不能进 RQ2（去混淆 / copy-bias）

RQ2 的代码 `experiments/experiment_deobfuscation_refineID.py` 做的事是：

1. 把多个不同标识符的所有 occurrence 用 `<|mask|>` **就地替换**在原 Java 序列里；
2. 调 `model.diffusion_generate(...)` 让模型**在这些原位 mask 上**一次性 denoise 出名字；
3. 对比两种上下文——"all-masked"（所有标识符都 mask）vs "target-only"（只 mask 目标，
   其它标识符保留原名）——来量化**模型抄袭周围名字的偏置（copy-bias）**。

DiffusionGemma 在这三步上都卡住：

- **第 1、2 步做不了**：它不能就地 infill `<|mask|>`，只能在 prompt 后面新开 canvas 生成。
  原序列里那些 mask 位置对它来说**没有可填的"洞"**。
- **第 3 步无从设置**：RQ2 的核心是**操纵 in-place mask 上下文**（全 mask vs 只 mask 目标）。
  DiffusionGemma 根本没有 in-place mask 上下文可以操纵，所以"copy-bias"这个被测现象在它身上
  **无法定义、无法构造**。
- 唯一的"绕路"是用 RQ1 那套 prompted per-site 提问。但那是**另一种操作**，测的是"按提示说出一个名字"，
  **不是** RQ2 想测的"原位 denoising 时会不会抄旁边的名字"。把这种数字放进 RQ2 的表里属于
  **苹果对橘子**，会误导结论。

> 一句话：RQ2 测的是 **in-place 多 mask denoising 的 copy-bias**，DiffusionGemma 在架构上
> 不具备 in-place denoising，连实验条件都搭不起来。

---

## 为什么不能进 RQ3（内部状态线性探针）

RQ3 的代码 `experiments/rq3_probe/extract_probe_states.py` 做的事是：

1. 用 `<|mask|>` 就地 mask 一个标识符，调 `model.diffusion_generate(...)` 并抓取
   **denoising 每一步、每一层在 mask token 位置上的 residual-stream hidden state**；
2. 训练线性探针，看模型在"好名字 vs 坏名字"上的内部表征**沿深度/沿 denoising 步**如何演化。

整套设计有两个硬性前提，DiffusionGemma 都不满足：

- **前提 A：标识符占据固定、贯穿 denoising 的 in-place token 位置。**
  DiffusionGemma 的答案生成在**追加的 canvas** 里，不在标识符原位；denoising 作用于 canvas
  而非上下文里的标识符 site。所以"模型处理这个标识符时的内部状态"**没有一个明确、可探针的位置**。
- **前提 B：可读的逐层、逐步 hidden-state 接口。**
  DiffusionGemma 是 128-expert MoE（层语义异构、有路由），并且跑在**另一套 transformers（5.11
  vs 主环境钉死的 4.57.1）**上，hidden-state 接口与 `AutoModel + diffusion_generate` 的钩子
  **完全不通用**。现有探针抽取代码无法迁移。

此外，论文本来就把 RQ3 限定在 **DiffuCoder / DreamCoder（fixed-canvas dLLM）**，正是因为它们
提供"稳定、in-place、可探针"的接口。塞进一个 block-AR MoE 会**改变实验语义且不可比**。

> 一句话：RQ3 探的是 **denoising 时标识符原位的内部表征轨迹**；DiffusionGemma 没有原位标识符
> 表征，也没有兼容的 hidden-state 接口，探针无处可探。

---

## 论文里怎么写（建议）

- **RQ1**：DiffusionGemma-26B-A4B 作为**第四个 dLLM**（Type = `dLLM-BlockAR`）进入
  Table II / 热力图 / CIS。加脚注：*"DiffusionGemma 采用 prompted per-site naming（block-AR，
  无 in-place infill，无 FIM base），与基模型的 FIM 行不直接可比，仅作为 dLLM 端任务数据点。"*
- **RQ2 / RQ3**：在 threats-to-validity 或方法节加一句：*"DiffusionGemma 为 block-autoregressive
  扩散模型，无法进行 in-place mask 去混淆（RQ2）与原位标识符的逐层 denoising 探针（RQ3），故这两个
  机制性实验仍限定在 fixed-canvas dLLM（DiffuCoder / DreamCoder）。"*

这样既满足"加入 DiffusionGemma 结果"的要求，又在机制实验上保持了科学上的可比性与诚实性。
