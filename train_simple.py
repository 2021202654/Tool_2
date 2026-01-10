import pandas as pd
import numpy as np
import xgboost as xgb
import json  # <--- 【新增 1】引入 json 库
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ================= 配置区域 =================
DATA_PATH = "data.csv"           # 请确认文件名
TARGET_COL = "thermal_conductivity" 
# 建议排除 "data_source" 防止过拟合文献来源
exclude_cols = ["id", "name", "data_source"]    
# ===========================================

def train_simple_model():
    print("1. 正在读取数据...")
    # 简单的文件读取容错
    try:
        if DATA_PATH.endswith('.xlsx'):
            df = pd.read_excel(DATA_PATH)
        else:
            df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {DATA_PATH}。请确保 CSV 文件在同一目录下。")
        return

    # 准备特征
    cols_to_drop = [TARGET_COL] + [c for c in exclude_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]
    
    print(f"原始特征列表: {list(X.columns)}")

    # 自动处理分类变量 (One-Hot Encoding)
    print("正在将文本特征转换为数值编码...")
    X = pd.get_dummies(X) # <--- 这里会产生额外的列
    print(f"编码后特征数量: {X.shape[1]}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n2. 正在训练模型...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # === 模型评估 ===
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    print(f"\n=== 模型评估 ===")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"R2 Score (拟合度): {r2:.4f}")

    # === 保存模型 ===
    model.save_model("simple_model.json")
    print("\nSUCCESS: 模型权重已保存为 'simple_model.json'")
    
    # === 【新增 2】关键步骤：保存特征名称列表 ===
    # 我们把最终训练用的所有列名（包括 One-Hot 产生的新列）存下来
    feature_names = list(X.columns)
    
    with open("model_features.json", "w", encoding='utf-8') as f:
        json.dump(feature_names, f, ensure_ascii=False)
        
    print("SUCCESS: 特征名称列表已保存为 'model_features.json' (这对 Agent 至关重要)")

if __name__ == "__main__":
    train_simple_model()