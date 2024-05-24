"""
@FileName：svm_model.py
@Description：机器学习
@Author：Surely
@Time： 2024 - 05 - 2024/5/21
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 忽略警告信息
import warnings
warnings.filterwarnings("ignore")

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据集
data = pd.read_csv("./diabetes_data_upload1.csv")

# 显示数据集的前几行
print(data.head())

# 数据预处理
# 将类别型特征转换为数值型
label_encoders = {}
categorical_columns = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                       'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
                       'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 选择相关性高的特征
selected_features = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'Polyphagia', 'Irritability', 'partial paresis']

# 标准化特征
scaler = StandardScaler()
data[selected_features] = scaler.fit_transform(data[selected_features])

# 显示处理后的数据集
print(data[selected_features].head())

# 分割特征和标签
X = data[selected_features]
y = data['class']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 模型训练和参数调整
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

svc = SVC()
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 最佳模型
best_model = grid_search.best_estimator_
print("最佳参数：", grid_search.best_params_)

# 模型预测
y_pred = best_model.predict(X_test)

# 模型评估
print("准确率: ", accuracy_score(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

# 可视化混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
