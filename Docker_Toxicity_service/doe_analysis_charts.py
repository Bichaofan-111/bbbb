import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np

# 设置绘图风格
sns.set_theme(style="whitegrid")
# 尝试设置中文字体，如果失败则回退到默认
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
FILE_NAME = "doe_fast_results.csv"

try:
    df = pd.read_csv(FILE_NAME)
    print(f"✅ 成功读取数据: {len(df)} 行")
except FileNotFoundError:
    print(f"❌ 找不到文件 {FILE_NAME}。正在生成模拟数据...")
    data = {
        'learning_rate': np.tile([2e-5, 5e-5, 2e-5, 5e-5], 5),
        'per_device_train_batch_size': np.tile([16, 16, 32, 32], 5),
        'roc_auc': np.random.uniform(0.90, 0.98, 20)
    }
    df = pd.DataFrame(data)

# 确保列名格式正确
if 'per_device_train_batch_size' in df.columns:
    df.rename(columns={'per_device_train_batch_size': 'batch_size'}, inplace=True)

# --- 核心统计分析 ---
formula = 'roc_auc ~ C(learning_rate) * C(batch_size)'
model = ols(formula, data=df).fit()

# --- 图表 1: ANOVA 方差分析表 ---
# 计算 ANOVA
anova_table = anova_lm(model, typ=2)

# 处理数据以生成表格
anova_display = anova_table.copy()

# 1. 计算 Mean Square (均方) = Sum_Sq / df
anova_display['mean_sq'] = anova_display['sum_sq'] / anova_display['df']

# 2. 重新排列列顺序，确保与表头对应 (Sum Sq, df, Mean Sq, F, Sig)
# 注意：anova_lm 输出的列名通常是 sum_sq, df, F, PR(>F)
anova_display = anova_display[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]

# 3. 格式化数值 - 关键修改：使用科学计数法显示微小数值
# 如果数值非常小，使用科学计数法，否则保留4位小数
def format_small_number(x):
    if x == 0: return "0"
    if abs(x) < 0.0001:
        return '{:.2e}'.format(x) # 科学计数法，例如 1.23e-05
    return '{:.4f}'.format(x)

anova_display['sum_sq'] = anova_display['sum_sq'].apply(format_small_number)
anova_display['mean_sq'] = anova_display['mean_sq'].apply(format_small_number)

anova_display['df'] = anova_display['df'].astype(int)
anova_display['F'] = anova_display['F'].map('{:.2f}'.format)

def format_p_value(x):
    if pd.isna(x): return ""
    if x < 0.001: return "<.001"
    return '{:.3f}'.format(x)

anova_display['PR(>F)'] = anova_display['PR(>F)'].apply(format_p_value)

# 4. 重命名列 (现在是5列对应5个名字)
anova_display.columns = ['Sum of Squares', 'df', 'Mean Square', 'F', 'Sig.']

# 绘制表格
fig_table = plt.figure(figsize=(10, 4))
ax_table = fig_table.add_subplot(111)
ax_table.axis('off')
ax_table.set_title("Statistical Significance of Factors (ANOVA)", fontsize=14, pad=20)

table = ax_table.table(cellText=anova_display.values,
                       colLabels=anova_display.columns,
                       rowLabels=anova_display.index,
                       loc='center',
                       cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# 高亮显著性 (P < 0.05)
for (row, col), cell in table.get_celld().items():
    if row > 0 and col == 4: # 第4列是 Sig.
        text = cell.get_text().get_text()
        if '<' in text or (text and float(text) < 0.05):
            cell.set_text_props(color='red', weight='bold')

plt.tight_layout()
plt.savefig("chart_1_anova_table.png", dpi=300, bbox_inches='tight')
print("📊 图表 1 已保存: ANOVA 表")

# --- 图表 2: 交互作用图 ---
plt.figure(figsize=(8, 6))
sns.pointplot(data=df, x="batch_size", y="roc_auc", hue="learning_rate",
              dodge=True, markers=['o', 's'], capsize=.1, errorbar='sd', linestyle='-')
plt.title("Estimated Marginal Means of ROC-AUC", fontsize=14)
plt.ylabel("Mean ROC-AUC Score")
plt.xlabel("Batch Size")
plt.legend(title="Learning Rate")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("chart_2_interaction_plot.png", dpi=300)
print("📈 图表 2 已保存: 交互作用图")

# --- 图表 3: 残差 Q-Q 图 ---
residuals = model.resid
fig_qq = plt.figure(figsize=(8, 6))
ax_qq = fig_qq.add_subplot(111)
sm.qqplot(residuals, line='s', ax=ax_qq, fit=True, markerfacecolor='skyblue', markeredgecolor='b', alpha=0.6)
ax_qq.set_title("Normal Q-Q Plot of Residuals", fontsize=14)
plt.tight_layout()
plt.savefig("chart_3_qq_plot.png", dpi=300)
print("📉 图表 3 已保存: 残差 Q-Q 图")

plt.show()