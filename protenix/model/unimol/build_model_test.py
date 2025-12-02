import argparse
import os
import torch
from unicore.data import Dictionary
from models.unimol import UniMolModel

# 假设上面的 UniMolModel 类定义已经存在，或者从文件中导入
# from models.unimol import UniMolModel 

def build_default_unimol_model():
    # 1. 构造 args 对象
    # 我们使用 Namespace 来模拟命令行参数传递
    args = argparse.Namespace()

    # 设置数据路径 (你需要修改为你实际的路径)
    args.data = "./data" 

    # -------------------------------------------------------
    # 关键点：补充 base_architecture 未覆盖但必须的参数
    # -------------------------------------------------------
    # 虽然 base_architecture 会填充大部分模型超参，
    # 但 'mode' 参数是在 add_args 中定义的，且在 forward 中会被用到。
    # 因为我们跳过了命令行解析，必须手动设置它。
    args.mode = "train"  # 或者 "infer"

    # 2. 加载字典 (根据你的要求)
    dict_path = os.path.join(args.data, "dict_mol.txt")
    
    # 检查文件是否存在，避免报错
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"找不到字典文件: {dict_path}，请确保路径正确。")
        
    mol_dictionary = Dictionary.load(dict_path)

    # 3. 实例化模型
    # 这里的关键是：UniMolModel 的 __init__ 会执行 base_architecture(args)
    # 它会自动检测 args 中缺失的属性，并填入默认值 (如 layers=15, dim=512)
    model = UniMolModel(args, mol_dictionary)

    return model, args

# --- 测试调用 ---
if __name__ == "__main__":
    try:
        model, args = build_default_unimol_model()
        
        print("✅ 模型构建成功")
        print("-" * 30)
        print(f"默认层数 (Encoder Layers): {args.encoder_layers}")
        print(f"默认维度 (Embed Dim): {args.encoder_embed_dim}")
        print(f"Attention Heads: {args.encoder_attention_heads}")
        print("-" * 30)
        
        # 简单测试一下 forward (需要构造假的输入数据)
        # 注意：实际输入需要符合 UniMol 的输入格式
        print(model)
        
    except Exception as e:
        print(f"构建失败: {e}")