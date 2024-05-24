import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
data = pd.read_csv("G:\\早期糖尿病预测\\Early-stage-diabetes-risk-prediction-main\\diabetes_data_upload1.csv")

# 数据预处理
# 将类别型特征转换为数值型
label_encoders = {}
categorical_columns = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                       'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
                       'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 标准化年龄特征
scaler = StandardScaler()
data['Age'] = scaler.fit_transform(data[['Age']])

# 分割特征和标签
X = data.drop(columns=['class'])
y = data['class']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 模型训练和参数调整
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 计算特征重要性
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': grid_search.best_estimator_.feature_importances_})

# 计算特征与目标的相关性
correlation_matrix = data.corr()

# 创建热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('特征与目标的相关性热力图')
plt.show()

# 打印特征重要性
print('特征重要性:')
print(feature_importances.sort_values(by='importance', ascending=False))