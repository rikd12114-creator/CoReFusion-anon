"""
实验A数据分析脚本：验证Edit Signal假设
分析 Layer-wise 和 Diffusion Step-wise 的表征偏移
"""
import pandas as pd
import numpy as np
from scipy import stats

# 读取数据
layer_df = pd.read_csv('results/edit_signal/edit_signals_layer_wise.csv')
diff_df = pd.read_csv('results/edit_signal/edit_signals_diffusion_steps.csv')

print("=" * 60)
print("实验A：Edit Signal 假设验证分析报告")
print("=" * 60)

# ============ Layer-wise 分析 ============
print("\n【1】Layer-wise 分析 (跨层表征偏移)")
print("-" * 50)

# 提取层号
layer_df['layer_num'] = layer_df['step'].str.extract(r'layer_(\d+)').astype(int)

# 按层聚合统计
layer_stats = layer_df.groupby('layer_num').agg({
    'cosine_sim': ['mean', 'std'],
    'l2_dist': ['mean', 'std']
}).round(4)

print("\n各层统计摘要:")
print(layer_stats)

# 计算早期层 vs 深层的差异
early_layers = layer_df[layer_df['layer_num'] <= 10]
deep_layers = layer_df[layer_df['layer_num'] >= 20]

print(f"\n早期层 (0-10) vs 深层 (20-28) 对比:")
print(f"  早期层 Cosine Sim: {early_layers['cosine_sim'].mean():.4f} ± {early_layers['cosine_sim'].std():.4f}")
print(f"  深层 Cosine Sim:   {deep_layers['cosine_sim'].mean():.4f} ± {deep_layers['cosine_sim'].std():.4f}")
print(f"  早期层 L2 Dist:    {early_layers['l2_dist'].mean():.2f} ± {early_layers['l2_dist'].std():.2f}")
print(f"  深层 L2 Dist:      {deep_layers['l2_dist'].mean():.2f} ± {deep_layers['l2_dist'].std():.2f}")

# 统计检验
t_cos, p_cos = stats.ttest_ind(early_layers['cosine_sim'], deep_layers['cosine_sim'])
t_l2, p_l2 = stats.ttest_ind(early_layers['l2_dist'], deep_layers['l2_dist'])

print(f"\n统计检验 (t-test):")
print(f"  Cosine Sim: t={t_cos:.3f}, p={p_cos:.2e} {'***' if p_cos < 0.001 else '**' if p_cos < 0.01 else '*' if p_cos < 0.05 else 'ns'}")
print(f"  L2 Dist:    t={t_l2:.3f}, p={p_l2:.2e} {'***' if p_l2 < 0.001 else '**' if p_l2 < 0.01 else '*' if p_l2 < 0.05 else 'ns'}")

# 计算层间趋势 (Spearman相关)
layer_means = layer_df.groupby('layer_num')[['cosine_sim', 'l2_dist']].mean()
rho_cos, p_rho_cos = stats.spearmanr(layer_means.index, layer_means['cosine_sim'])
rho_l2, p_rho_l2 = stats.spearmanr(layer_means.index, layer_means['l2_dist'])

print(f"\n层深度与指标的Spearman相关:")
print(f"  Layer vs Cosine Sim: ρ={rho_cos:.4f}, p={p_rho_cos:.2e}")
print(f"  Layer vs L2 Dist:    ρ={rho_l2:.4f}, p={p_rho_l2:.2e}")

# ============ Diffusion Step-wise 分析 ============
print("\n" + "=" * 60)
print("【2】Diffusion Step-wise 分析 (跨降噪步表征演变)")
print("-" * 50)

# 提取步数
diff_df['step_num'] = diff_df['step'].str.extract(r'diff_step_(\d+)').astype(int)

# 按步数聚合
step_stats = diff_df.groupby('step_num').agg({
    'cosine_sim': ['mean', 'std'],
    'l2_dist': ['mean', 'std']
}).round(4)

print("\n各降噪步统计摘要 (部分):")
print(step_stats.iloc[::5])  # 每5步显示一次

# 早期步 vs 后期步
early_steps = diff_df[diff_df['step_num'] <= 10]
late_steps = diff_df[diff_df['step_num'] >= 25]

print(f"\n早期步 (0-10) vs 后期步 (25-31) 对比:")
print(f"  早期步 Cosine Sim: {early_steps['cosine_sim'].mean():.4f} ± {early_steps['cosine_sim'].std():.4f}")
print(f"  后期步 Cosine Sim: {late_steps['cosine_sim'].mean():.4f} ± {late_steps['cosine_sim'].std():.4f}")
print(f"  早期步 L2 Dist:    {early_steps['l2_dist'].mean():.2f} ± {early_steps['l2_dist'].std():.2f}")
print(f"  后期步 L2 Dist:    {late_steps['l2_dist'].mean():.2f} ± {late_steps['l2_dist'].std():.2f}")

# 统计检验
t_cos_d, p_cos_d = stats.ttest_ind(early_steps['cosine_sim'], late_steps['cosine_sim'])
t_l2_d, p_l2_d = stats.ttest_ind(early_steps['l2_dist'], late_steps['l2_dist'])

print(f"\n统计检验 (t-test):")
print(f"  Cosine Sim: t={t_cos_d:.3f}, p={p_cos_d:.2e} {'***' if p_cos_d < 0.001 else '**' if p_cos_d < 0.01 else '*' if p_cos_d < 0.05 else 'ns'}")
print(f"  L2 Dist:    t={t_l2_d:.3f}, p={p_l2_d:.2e} {'***' if p_l2_d < 0.001 else '**' if p_l2_d < 0.01 else '*' if p_l2_d < 0.05 else 'ns'}")

# 计算步数趋势
step_means = diff_df.groupby('step_num')[['cosine_sim', 'l2_dist']].mean()
rho_cos_d, p_rho_cos_d = stats.spearmanr(step_means.index, step_means['cosine_sim'])
rho_l2_d, p_rho_l2_d = stats.spearmanr(step_means.index, step_means['l2_dist'])

print(f"\n降噪步数与指标的Spearman相关:")
print(f"  Step vs Cosine Sim: ρ={rho_cos_d:.4f}, p={p_rho_cos_d:.2e}")
print(f"  Step vs L2 Dist:    ρ={rho_l2_d:.4f}, p={p_rho_l2_d:.2e}")

# ============ 假设验证总结 ============
print("\n" + "=" * 60)
print("【3】假设验证总结")
print("=" * 60)

print("""
实验A假设：模型在提取 Smell Token 与 GT Token 时，会在深度和降噪步数上
          体现出明确的表征偏移 (Edit Signal)。

验证结果：
""")

# Layer-wise 结论
if rho_cos > 0.5 and p_rho_cos < 0.05:
    layer_cos_result = "✓ 支持假设"
    layer_cos_detail = f"随层深度增加，Cosine相似度显著上升 (ρ={rho_cos:.3f})"
else:
    layer_cos_result = "✗ 不支持"
    layer_cos_detail = f"层深度与Cosine相似度无显著正相关 (ρ={rho_cos:.3f})"

if rho_l2 > 0.5 and p_rho_l2 < 0.05:
    layer_l2_result = "✓ 支持假设"
    layer_l2_detail = f"随层深度增加，L2距离显著增大 (ρ={rho_l2:.3f})"
else:
    layer_l2_result = "✗ 不支持"
    layer_l2_detail = f"层深度与L2距离无显著正相关 (ρ={rho_l2:.3f})"

print(f"[Layer-wise]")
print(f"  Cosine Sim趋势: {layer_cos_result} - {layer_cos_detail}")
print(f"  L2 Dist趋势:    {layer_l2_result} - {layer_l2_detail}")

# Diffusion step-wise 结论
if rho_cos_d > 0.5 and p_rho_cos_d < 0.05:
    diff_cos_result = "✓ 支持假设"
    diff_cos_detail = f"随降噪步数增加，Cosine相似度显著上升 (ρ={rho_cos_d:.3f})"
else:
    diff_cos_result = "✗ 不支持"
    diff_cos_detail = f"降噪步数与Cosine相似度无显著正相关 (ρ={rho_cos_d:.3f})"

if rho_l2_d < -0.5 and p_rho_l2_d < 0.05:
    diff_l2_result = "✓ 支持假设"
    diff_l2_detail = f"随降噪步数增加，L2距离显著减小 (ρ={rho_l2_d:.3f})"
else:
    diff_l2_result = "✗ 不支持"
    diff_l2_detail = f"降噪步数与L2距离无显著负相关 (ρ={rho_l2_d:.3f})"

print(f"\n[Diffusion Step-wise]")
print(f"  Cosine Sim趋势: {diff_cos_result} - {diff_cos_detail}")
print(f"  L2 Dist趋势:    {diff_l2_result} - {diff_l2_detail}")

# 总体结论
print("\n" + "-" * 50)
print("总体结论:")
support_count = sum([
    rho_cos > 0.5 and p_rho_cos < 0.05,
    rho_l2 > 0.5 and p_rho_l2 < 0.05,
    rho_cos_d > 0.5 and p_rho_cos_d < 0.05,
    rho_l2_d < -0.5 and p_rho_l2_d < 0.05
])

if support_count >= 3:
    print(f"  ★ 假设得到强支持 ({support_count}/4 指标符合预期)")
elif support_count >= 2:
    print(f"  ◆ 假设得到部分支持 ({support_count}/4 指标符合预期)")
else:
    print(f"  ○ 假设未得到充分支持 ({support_count}/4 指标符合预期)")

# 数据规模
n_samples = layer_df['sample_id'].nunique()
n_layers = layer_df['layer_num'].nunique()
n_steps = diff_df['step_num'].nunique()
print(f"\n数据规模: {n_samples} 样本 × {n_layers} 层 / {n_steps} 降噪步")
