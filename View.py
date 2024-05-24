import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 读取数据集
data = pd.read_csv("G:\\早期糖尿病预测\\Early-stage-diabetes-risk-prediction-main\\diabetes_data_upload.csv")

# 设置字体和编码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# # 计算Polyuria中不同取值对应的class值为Positive和Negative的样本数量
# polyuria_counts = data.groupby(['Polyuria', 'class']).size().unstack()
# # 绘制柱状图
#
#
# polyuria_counts.plot(kind='bar', stacked=True)
# plt.title('多尿 vs 糖尿病')
# plt.xlabel('多尿')
# plt.ylabel('数量')
# plt.xticks(rotation=0)  # 保持x轴标签水平
# plt.legend(title='Class')
# plt.show()

####性别和患病关系
# 计算Gender中不同取值对应的class值为Positive和Negative的样本数量
# polyuria_counts = data.groupby(['Gender', 'class']).size().unstack()
# # 绘制柱状图
#
#
# polyuria_counts.plot(kind='bar', stacked=True)
# plt.title('性别 vs 糖尿病')
# plt.xlabel('性别')
# plt.ylabel('数量')
# plt.xticks(rotation=0)  # 保持x轴标签水平
# plt.legend(title='Class')
# plt.show()

# # 计算Obesity中不同取值对应的class值为Positive和Negative的样本数量
# polyuria_counts = data.groupby(['Obesity', 'class']).size().unstack()
# # 绘制柱状图
#
#
# polyuria_counts.plot(kind='bar', stacked=True)
# plt.title('肥胖 vs 糖尿病')
# plt.xlabel('肥胖')
# plt.ylabel('数量')
# plt.xticks(rotation=0)  # 保持x轴标签水平
# plt.legend(title='Class')
# plt.show()


# 计算Obesity中不同取值对应的class值为Positive和Negative的样本数量
polyuria_counts = data.groupby(['visual blurring', 'class']).size().unstack()
# 绘制柱状图


polyuria_counts.plot(kind='bar', stacked=True)
plt.title('视力模糊 vs 糖尿病')
plt.xlabel('视力模糊')
plt.ylabel('数量')
plt.xticks(rotation=0)  # 保持x轴标签水平
plt.legend(title='Class')
plt.show()