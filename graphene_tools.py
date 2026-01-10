import json
import numpy as np
import pandas as pd
import xgboost as xgb
from langchain.tools import tool

# === 全局配置 ===
MODEL_PATH = "simple_model.json"
FEATURE_PATH = "model_features.json"

# === 全局缓存变量 (单例模式) ===
_xgb_model = None
_model_features = None

def load_resources():
    """
    加载模型和特征列表。
    使用全局变量缓存，避免每次调用工具都重新读取文件。
    """
    global _xgb_model, _model_features
    
    # 1. 加载特征名称列表 (这是对齐的关键)
    if _model_features is None:
        try:
            with open(FEATURE_PATH, "r", encoding='utf-8') as f:
                _model_features = json.load(f)
        except FileNotFoundError:
            return None, None, f"找不到 {FEATURE_PATH}，请先运行 train_simple.py"

    # 2. 加载 XGBoost 模型
    if _xgb_model is None:
        _xgb_model = xgb.Booster()
        try:
            _xgb_model.load_model(MODEL_PATH)
        except Exception as e:
            return None, None, f"模型加载失败: {str(e)}"
            
    return _xgb_model, _model_features, ""

@tool
def ml_prediction_tool(length_um: float, temperature_k: float, defect_ratio: float) -> str:
    """
    [机器学习工具] 基于微观结构预测石墨烯热导率。
    当用户询问预测具体数值时调用。
    
    参数:
    - length_um: 样品长度 (um)，默认建议 10.0
    - temperature_k: 温度 (K)
    - defect_ratio: 缺陷率 (例如 0.01 表示 1%)
    """
    # 1. 加载资源
    model, features, error_msg = load_resources()
    if error_msg:
        return f"系统错误: {error_msg}"

    try:
        # === 核心逻辑：特征对齐 ===
        
        # A. 创建一个只有 1 行的 DataFrame，列名与训练时完全一致，初始值全填 0
        input_df = pd.DataFrame(0, index=[0], columns=features)
        
        # B. 填入用户提供的参数
        # 注意：这里假设你的 CSV 训练数据中，列名包含 "length", "temperature", "defect_ratio"
        # 如果你的 CSV 列名是中文 (如 "长度")，请在这里修改映射，例如: input_df['长度'] = length_um
        
        # 尝试匹配常见的列名写法，增强容错性
        if 'length_um' in features: input_df['length_um'] = length_um
        elif 'length' in features: input_df['length'] = length_um
            
        if 'temperature_k' in features: input_df['temperature_k'] = temperature_k
        elif 'temperature' in features: input_df['temperature'] = temperature_k
            
        if 'defect_ratio' in features: input_df['defect_ratio'] = defect_ratio
        elif 'defect' in features: input_df['defect'] = defect_ratio

        # C. 转换为 DMatrix (XGBoost 专用格式)
        dtrain = xgb.DMatrix(input_df)
        
        # D. 执行预测
        pred_value = model.predict(dtrain)[0]
        
        # E. 格式化输出
        return f"{pred_value:.2f} W/mK"
        
    except Exception as e:
        return f"预测计算过程出错: {str(e)}"

@tool
def physics_calculation_tool(temperature_k: float, defect_ratio: float) -> str:
    """
    [物理公式工具] 使用 Klemens-Callaway 模型计算理论热导率上限。
    当预测结果异常或需要理论基准时使用。
    """
    try:
        if temperature_k <= 0: return "错误：温度必须大于 0K"
        
        # 简化的物理模型模拟 (K ~ 1/T)
        # 实际科研中这里应该替换为真实的积分公式
        A = 1.5e-4
        B = 2.0e3
        
        # 避免除以 0
        denominator = (A * temperature_k + B * defect_ratio)
        if denominator == 0: denominator = 1e-9
            
        kappa_theory = 1 / denominator
        
        # 简单修正量级用于演示
        kappa_theory = kappa_theory * 1000 
        
        return f"{kappa_theory:.2f} W/mK (理论值)"
    except Exception as e:
        return f"物理计算出错: {str(e)}"