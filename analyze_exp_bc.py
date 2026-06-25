"""
实验B & C 数据分析脚本：验证 Unmasking Order 和 Token Ranking 假设
"""
import pandas as pd
import numpy as np
from scipy import stats

# 读取数据
unmask_df = pd.read_csv('results/abc/unmasking_order_20260312_225458.csv')
ranking_df = pd.read_csv('results/abc/token_ranking_20260312_230225.csv')

print("=" * 70)
print("实验B & C：Unmasking Order 和 Token Ranking 假设验证分析报告")
print("=" * 70)

# ============ 实验B：Unmasking Order 分析 ============
print("\n" + "=" * 70)
print("【实验B】Unmasking Order 分析 (Smell Token 是否更早被确定)")
print("=" * 70)

print(f"\n数据规模: {len(unmask_df)} 条记录, {unmask_df['sample_id'].nunique()} 个样本")

# 分组统计
smell_tokens = unmask_df[unmask_df['is_smell_token'] == True]
context_tokens = unmask_df[unmask_df['is_smell_token'] == False]

print(f"\nSmell Tokens: {len(smell_tokens)} 个")
print(f"Context Tokens: {len(context_tokens)} 个")

# 核心指标对比
print("\n" + "-" * 50)
print("核心指标对比:")
print("-" * 50)

# avg_flip_step 分析
smell_flip = smell_tokens['avg_flip_step'].dropna()
context_flip = context_tokens['avg_flip_step'].dropna()

print(f"\n[avg_flip_step] (平均翻牌步数，越小=越早确定)")
print(f"  Smell Tokens:   {smell_flip.mean():.2f} ± {smell_flip.std():.2f} (median: {smell_flip.median():.1f})")
print(f"  Context Tokens: {context_flip.mean():.2f} ± {context_flip.std():.2f} (median: {context_flip.median():.1f})")

# first_confident_step 分析
smell_conf = smell_tokens['first_confident_step'].dropna()
context_conf = context_tokens['first_confident_step'].dropna()

print(f"\n[first_confident_step] (首次高置信步数，越小=越早确定)")
print(f"  Smell Tokens:   {smell_conf.mean():.2f} ± {smell_conf.std():.2f} (median: {smell_conf.median():.1f})")
print(f"  Context Tokens: {context_conf.mean():.2f} ± {context_conf.std():.2f} (median: {context_conf.median():.1f})")

# 统计检验
print("\n" + "-" * 50)
print("统计检验:")
print("-" * 50)

# Mann-Whitney U 检验 (非参数检验，更稳健)
u_flip, p_flip = stats.mannwhitneyu(smell_flip, context_flip, alternative='less')
u_conf, p_conf = stats.mannwhitneyu(smell_conf, context_conf, alternative='less')

print(f"\nMann-Whitney U 检验 (单侧: Smell < Context):")
print(f"  avg_flip_step:       U={u_flip:.0f}, p={p_flip:.4e} {'***' if p_flip < 0.001 else '**' if p_flip < 0.01 else '*' if p_flip < 0.05 else 'ns'}")
print(f"  first_confident_step: U={u_conf:.0f}, p={p_conf:.4e} {'***' if p_conf < 0.001 else '**' if p_conf < 0.01 else '*' if p_conf < 0.05 else 'ns'}")

# 效应量 (Cohen's d)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

d_flip = cohens_d(smell_flip, context_flip)
d_conf = cohens_d(smell_conf, context_conf)

print(f"\n效应量 (Cohen's d, 负值表示Smell更小/更早):")
print(f"  avg_flip_step:       d={d_flip:.3f} ({'大' if abs(d_flip) > 0.8 else '中' if abs(d_flip) > 0.5 else '小'}效应)")
print(f"  first_confident_step: d={d_conf:.3f} ({'大' if abs(d_conf) > 0.8 else '中' if abs(d_conf) > 0.5 else '小'}效应)")

# 假设验证
print("\n" + "-" * 50)
print("实验B假设验证:")
print("-" * 50)
print("""
假设：Smell Token 作为语义症结，会比 Context Token 更早被模型确定（Flip）。
      即 Smell 的 avg_flip_step 和 first_confident_step 应显著小于 Context。
""")

b_flip_support = p_flip < 0.05 and smell_flip.mean() < context_flip.mean()
b_conf_support = p_conf < 0.05 and smell_conf.mean() < context_conf.mean()

if b_flip_support:
    print(f"  avg_flip_step:       ✓ 支持假设 (Smell {smell_flip.mean():.1f} < Context {context_flip.mean():.1f}, p={p_flip:.2e})")
else:
    print(f"  avg_flip_step:       ✗ 不支持 (Smell {smell_flip.mean():.1f} vs Context {context_flip.mean():.1f}, p={p_flip:.2e})")

if b_conf_support:
    print(f"  first_confident_step: ✓ 支持假设 (Smell {smell_conf.mean():.1f} < Context {context_conf.mean():.1f}, p={p_conf:.2e})")
else:
    print(f"  first_confident_step: ✗ 不支持 (Smell {smell_conf.mean():.1f} vs Context {context_conf.mean():.1f}, p={p_conf:.2e})")


# ============ 实验C：Token Ranking / Entropy 分析 ============
print("\n" + "=" * 70)
print("【实验C】Token Ranking 分析 (Smell Token 的熵变化是否更剧烈)")
print("=" * 70)

print(f"\n数据规模: {len(ranking_df)} 条记录, {ranking_df['sample_id'].nunique()} 个样本")

# 分组统计
smell_rank = ranking_df[ranking_df['is_smell_token'] == True]
context_rank = ranking_df[ranking_df['is_smell_token'] == False]

print(f"\nSmell Tokens: {len(smell_rank)} 个")
print(f"Context Tokens: {len(context_rank)} 个")

# 核心指标对比
print("\n" + "-" * 50)
print("核心指标对比:")
print("-" * 50)

# mean_entropy_change 分析
smell_entropy = smell_rank['mean_entropy_change'].dropna()
context_entropy = context_rank['mean_entropy_change'].dropna()

# 过滤掉0值（可能是未参与计算的token）
smell_entropy_nonzero = smell_entropy[smell_entropy > 0]
context_entropy_nonzero = context_entropy[context_entropy > 0]

print(f"\n[mean_entropy_change] (平均熵变化，越大=预测越不稳定/摇摆)")
print(f"  Smell Tokens:   {smell_entropy.mean():.4f} ± {smell_entropy.std():.4f} (median: {smell_entropy.median():.4f})")
print(f"  Context Tokens: {context_entropy.mean():.4f} ± {context_entropy.std():.4f} (median: {context_entropy.median():.4f})")

print(f"\n[mean_entropy_change > 0] (排除未参与计算的token)")
print(f"  Smell Tokens:   {smell_entropy_nonzero.mean():.4f} ± {smell_entropy_nonzero.std():.4f} (n={len(smell_entropy_nonzero)})")
print(f"  Context Tokens: {context_entropy_nonzero.mean():.4f} ± {context_entropy_nonzero.std():.4f} (n={len(context_entropy_nonzero)})")

# max_entropy_change 分析
smell_max_ent = smell_rank['max_entropy_change'].dropna()
context_max_ent = context_rank['max_entropy_change'].dropna()

print(f"\n[max_entropy_change] (最大熵跳跃，越大=存在剧烈洗牌)")
print(f"  Smell Tokens:   {smell_max_ent.mean():.4f} ± {smell_max_ent.std():.4f} (median: {smell_max_ent.median():.4f})")
print(f"  Context Tokens: {context_max_ent.mean():.4f} ± {context_max_ent.std():.4f} (median: {context_max_ent.median():.4f})")

# 统计检验
print("\n" + "-" * 50)
print("统计检验:")
print("-" * 50)

# Mann-Whitney U 检验 (单侧: Smell > Context)
u_ent, p_ent = stats.mannwhitneyu(smell_entropy, context_entropy, alternative='greater')
u_max, p_max = stats.mannwhitneyu(smell_max_ent, context_max_ent, alternative='greater')

print(f"\nMann-Whitney U 检验 (单侧: Smell > Context):")
print(f"  mean_entropy_change: U={u_ent:.0f}, p={p_ent:.4e} {'***' if p_ent < 0.001 else '**' if p_ent < 0.01 else '*' if p_ent < 0.05 else 'ns'}")
print(f"  max_entropy_change:  U={u_max:.0f}, p={p_max:.4e} {'***' if p_max < 0.001 else '**' if p_max < 0.01 else '*' if p_max < 0.05 else 'ns'}")

# 效应量
d_ent = cohens_d(smell_entropy, context_entropy)
d_max = cohens_d(smell_max_ent, context_max_ent)

print(f"\n效应量 (Cohen's d, 正值表示Smell更大):")
print(f"  mean_entropy_change: d={d_ent:.3f} ({'大' if abs(d_ent) > 0.8 else '中' if abs(d_ent) > 0.5 else '小'}效应)")
print(f"  max_entropy_change:  d={d_max:.3f} ({'大' if abs(d_max) > 0.8 else '中' if abs(d_max) > 0.5 else '小'}效应)")

# 假设验证
print("\n" + "-" * 50)
print("实验C假设验证:")
print("-" * 50)
print("""
假设：如果模型在 Smell 位置经历"摇摆和自我修正"，那么 Smell Token 的
      熵变化（|H_{k+1} - H_k|）应显著大于 Context Token。
""")

c_mean_support = p_ent < 0.05 and smell_entropy.mean() > context_entropy.mean()
c_max_support = p_max < 0.05 and smell_max_ent.mean() > context_max_ent.mean()

if c_mean_support:
    print(f"  mean_entropy_change: ✓ 支持假设 (Smell {smell_entropy.mean():.4f} > Context {context_entropy.mean():.4f}, p={p_ent:.2e})")
else:
    print(f"  mean_entropy_change: ✗ 不支持 (Smell {smell_entropy.mean():.4f} vs Context {context_entropy.mean():.4f}, p={p_ent:.2e})")

if c_max_support:
    print(f"  max_entropy_change:  ✓ 支持假设 (Smell {smell_max_ent.mean():.4f} > Context {context_max_ent.mean():.4f}, p={p_max:.2e})")
else:
    print(f"  max_entropy_change:  ✗ 不支持 (Smell {smell_max_ent.mean():.4f} vs Context {context_max_ent.mean():.4f}, p={p_max:.2e})")


# ============ 总体结论 ============
print("\n" + "=" * 70)
print("【总体结论】")
print("=" * 70)

results = {
    'B_flip': b_flip_support,
    'B_conf': b_conf_support,
    'C_mean': c_mean_support,
    'C_max': c_max_support
}

support_count = sum(results.values())

print(f"""
实验B (Unmasking Order):
  - avg_flip_step:        {'✓' if results['B_flip'] else '✗'}
  - first_confident_step: {'✓' if results['B_conf'] else '✗'}

实验C (Token Ranking / Entropy):
  - mean_entropy_change:  {'✓' if results['C_mean'] else '✗'}
  - max_entropy_change:   {'✓' if results['C_max'] else '✗'}

总计: {support_count}/4 指标支持假设
""")

if support_count >= 3:
    print("★ 实验B和C的假设得到强支持")
elif support_count >= 2:
    print("◆ 实验B和C的假设得到部分支持")
else:
    print("○ 实验B和C的假设未得到充分支持，需要进一步分析")

# 额外分析：检查数据分布
print("\n" + "-" * 50)
print("补充分析：数据分布特征")
print("-" * 50)

print(f"\nSmell Token 样本分布:")
print(f"  样本数: {len(smell_tokens)}")
if len(smell_tokens) > 0:
    print(f"  唯一标识符: {smell_tokens['identifier_name'].nunique()}")
    print(f"  示例: {smell_tokens['identifier_name'].head(10).tolist()}")

print(f"\nContext Token 样本分布:")
print(f"  样本数: {len(context_tokens)}")
print(f"  唯一标识符: {context_tokens['identifier_name'].nunique()}")
