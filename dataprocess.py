"""
@FileName：dataprocess.py\n
@Description：机器学习\n
@Author：Surely\n
@Time： 2024 - 05 - 2024/5/22\n
"""
import pandas as pd

data = pd.read_csv("G:\\早期糖尿病预测\\Early-stage-diabetes-risk-prediction-main\\diabetes_data_upload.csv")

data = data.sample(frac=1).reset_index(drop=True)

data.to_csv("G:\\早期糖尿病预测\\Early-stage-diabetes-risk-prediction-main\\diabetes_data_upload1.csv", index=False)