import argparse
import os
import torch
from unicore.data import Dictionary
from protenix.model.unimol.models import UniMolModel


def build_default_unimol_model(mode="train", dict_path="./data/"):
    # 1. 构造 args 对象
    # 我们使用 Namespace 来模拟命令行参数传递
    args = argparse.Namespace()

    args.data = dict_path
    args.mode = mode

    # 2. 加载字典 (根据你的要求)
    dict_path = os.path.join(args.data, "dict_mol.txt")
    
    # 检查文件是否存在，避免报错
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"找不到字典文件: {dict_path}，请确保路径正确。")
        
    mol_dictionary = Dictionary.load(dict_path)
    mask_idx = mol_dictionary.add_symbol("[MASK]", is_special=True)

    # 3. 实例化模型
    # 这里的关键是：UniMolModel 的 __init__ 会执行 base_architecture(args)
    # 它会自动检测 args 中缺失的属性，并填入默认值 (如 layers=15, dim=512)
    model = UniMolModel(args, mol_dictionary)

    return model