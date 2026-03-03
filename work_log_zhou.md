# 1.ESM 和 unimol 接入 protenix

整体上，我希望借助protenix和qbiolip的数据库进行训练，完成虚拟筛选的对比学习，我提供给你一些代码，请告诉我下面应该干什么？
esm 编码为[L,1280] 是残基级别的表示，需要进行mean聚合
unimol 编码为[L,512] 是原子级别的表示，
protenix 维度为258
我首先将pro通过esm编码，将lig通过unimol编码，然后投影到258维，进行对比学习，这个时候得到loss1，然后将生成结构形成protenix的损失loss2，将这两个损失一起进行训练
注意的是，我们需要冻结protenix，esm unimol，只训练我们的MLP投影层

1. 我是用16张卡训练，batchaseize = 4，所以是方案A ，修改batchsize ，
2. 投影层改成MLP，至少两层relu
3. 

# 2.Biolip 的数据整理

Number of entries for regular ligands: 501072

1. 筛选Biolip里面的pro+lig部分，用esm + unimol 进行对比学习，然后和protenix一起优化
2. biolip_nr.txt.gz 内部每一行是protein-ligand交互样本，解析注释文件，过滤ligand的金属离子，合并receptorPDB 和ligandPDB
   1. receptor:与配体相互作用的蛋白结构（按链切分）
   2. ligand：对应配体结构，每个binding site/serial一份
      官方说明了怎么从这个文件中获得PDB结构的数据
      然而，困难的是我无法从其中下载，所以我选择了Q-biolip，其中包括四级结构https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download/index_biolip.html
3. protenix 有实验用的预训练脚本，默认会做一堆过滤（去水、去氢、去异常链/原子等,如果你输入的 CIF 不是“标准从 RCSB 下载的原始 mmCIF”，而是你自己生成/裁剪的，通常建议加 -d：-d 会禁用这些 filters，并且不会尝试 expand 到 Assembly 1
   1. input:cif文件
   2. output: .pkl.gz 的“训练可用缓存”（包含 atom_array/token_array、序列、分辨率等信息;
   3. output: output_csv（indices），用于catalog chains 和 interfaces，训练时按这个采样
      以上，数据准备已经结束.
      wget -c https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/rec_biolip.tar.gz

# 3. Q-biolip 数据下载及处理

## 3.1 下载

https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download/
Interaction-based → Protein–small molecules interaction → Non-redundant

## 3.2 run to clean

我通过Q-biolip的数据进行通过ESM 和UNIMOL 改良的protenix的模型训练，具体是lig通过unimol prot通过esm，然后简单投影到258维度进行对比学习，过程中，一个batchsize = 4，用16张卡进行训练，得到对比学习的loss；同时我们的复合物prot-lig的结构进行protenix的结构损失形成loss2
我首先进行投影层的训练，冻结protenix，然后根据训练情况，是否冻结unimol和esm
现在我有两个文件：

- run_clean_qbiolip.sh
- clean_qbiolip.py
  实现两个过程，将下载的lig和prot结构进行清洗获得干净的训练数据（可以进入我修改的unimol+protenix+esm的模型），同时需要调用protenix的数处理py文件进行数据规范化。

```bash
cd /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif

# 关键：指定 Protenix 根目录（按你的路径）
export PROTENIX_ROOT=/home/dataset-local/tmp/zsl/Protenix

# CPU 并行（按机器调整）
export WORKERS=32
export PREP_WORKERS=32

# 输出目录（可改）
export OUT_ROOT=$PWD/clean_qbiolip_nonredund

# （可选）更严格的“小分子”过滤：只保留 ligand.json 里有 SMILES 的 ligid
# export REQUIRE_SMILES=1

# （可选）如果你数据包含大量 2024-06-08 之后的新 CCD code，建议打开 CCD 更新（需要联网）
# export UPDATE_CCD=1

bash run_clean_qbiolip_nonredund_mmcif.sh

```

### 3.2.1 location

[DONE] complexes: /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/complexes
[DONE] manifest:  /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/manifest.tsv
[DONE] failed:    /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/failed.tsv
[DONE] stats:     /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/stats.txt
[DONE] cif_list:  /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/complexes_cif_paths.txt
[DONE] indices:   /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/qbiolip_nonredund_indices.csv
[DONE] bioasm:    /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/qbiolip_nonredund_bioassembly

### 3.2.2 expliation

原始结构：受体-配体结构
annotations: csv ID PDB Assembly ligandID 位点 binding Site residues
ligand.json : smiles mw 小分子集合

1. 合并rec 和lig 形成complex cif
2. 清洗ligand 只保留一个model 去掉alt conf H water，选择最佳的ligand residue
3. 清洗rec：保留model 去掉alter conf /H/water
   调用protenix标准预处理脚本
   find complexes -name "*.cif" > complexes_cif_paths.txt

python3 $PROTENIX_ROOT/scripts/prepare_training_data.py -i `<list>` -o indices.csv -b bioassembly -n `<cpu>` -d

run_clean_qbiolip_nonredund_mmc…

这里用了 -d，含义是：
“这批 CIF 不是 RCSB 原生下载的 WeightedPDB 格式，不要再套一堆 WeightedPDB 过滤/assembly 扩展规则。”

# 模块整理

调用方式

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

python runner/train.py \
      --model_name "protenix_mini_esm650m_unimol_contrast_v0.2.0" \
      --run_name "test_run_final" \
      --base_dir "./output" \
      --model.N_cycle 1 \
      --sample_diffusion.N_sample 1 \
      --max_steps 10 \
      --log_interval 1 \
      --eval_interval 10 \
      --checkpoint_interval 0 \
      --use_wandb False \
      --project "test_debug"
```
## 如何将loss加入进行训练
qbiolip 输出为:
1. receptor 和 ligand 合成一个complex cif 输出到out_root/complect/../8cif
2. complex cif 输入到prepare_training_data.py 生成
   1. qbiolip_*_indices.csv
   2. qbiolip_*_bioassembly/*.pkl.gz
# 问题

1. batchsize=1是用ddp吗？ 不是，用16张卡，batchsize=4，先简单训练一下，看看可不可以跑通，如果不行，我们再调整一下状态
2. 数据是149w的结构数据 40w结构数据？直接下载mmcif的文件 有4.5w蛋白质，同时有13w小分子对应，进行数据清洗和合并，形成新的结构
3. 对比学习投影，我就只用了一个line投影到258维？不行，用MLP
4. 接下来是冻结unimol 和 esm protenix 直接训练 投影吗？不一定是冻结esm和unimol，反正protenix是一定要修改的，需要开放mlp和unimol，esm不一定要开放
5.
### 修改完成qbiolip(单卡)
export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && conda activate protenix311 && python -u runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name debug_qbiolip_single2 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 2 \
  --log_interval 1 \
  --eval_interval 10 \
  --checkpoint_interval 0 \
  --use_wandb False \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund

  ### q1 存在ligand不识别的问题?
1.   根据qbiolip 下载select ligand的黑名单
2.   python3 /home/dataset-local/tmp/zsl/Protenix/scripts/gen_ccd_cache.py -n 32 下载ccd数据
```bash
CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "ccd_cache","components.cif")
CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
    DATA_ROOT_DIR,"ccd_cache","components.cif.rdkit_mol.pkl"
)
```
3. 为了避免DDP过程中出现ligand=0的情况会导致NCCL Hang，需要对过程进行补充(解决方法：对缺少的部分进行补充，让他们ligand=0的部分也要进行同样的处理流程，但是loss不做贡献)

### qbiolip（四卡）
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name debug_qbiolip_ddp4 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 2 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb False \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund
#### q2 不同的rank走了不同的分支，有的有ligand 有的没有，导致某些需要同步梯度的参数在部分rank上变成了unused 从而梯度上死锁超时
find_unused_parameters: true
static_graph=False,# 原本是true
## 考虑到cropsize导致的ligand删除
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"], _, _,_= (




### wandb true
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name debug_qbiolip_ddp4 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 2 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb True \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund



  ## 直接训练的代码
  cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name qbiolip_PLonly_ddp4_bs1_e100 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 3236900 \
  --log_interval 10 \
  --eval_interval 999999999 \
  --checkpoint_interval 20000 \
  --use_wandb True \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund

### structure-only sanity check (Protenix loss without contrastive term)
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_structure_only_v0.1.0 \
  --run_name structure_only_protenix \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 50000 \
  --log_interval 10 \
  --eval_interval 999999999 \
  --checkpoint_interval 20000 \
  --use_wandb True \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund \
  --contrast.enable False

该命令复用了 `protenix_mini_structure_only_v0.1.0` 配置，它继承了现有的 ESM/UniMol 设置但把 contrast_loss 关闭，便于观察仅用 Protenix 结构损失的收敛行为。

## 本次修改完成（2026-03-03）

### 1) 运行方式修正
- 当前环境里 `conda activate` 不可用，统一改为 `conda run -n protenix311 ...`
- NCCL 异步报错变量改为 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

### 2) 纯 Protenix 训练（无对比损失）
- 已新增模型配置：`protenix_mini_structure_only_v0.1.0`
- 使用该模型时 `contrast.enable=False`，只优化结构相关损失（diffusion/distogram/confidence）

### 3) 联合训练路径代码对齐（核心改动）
- 文件：`protenix/model/protenix.py`
- 已将同一套 ESM/UniMol MLP 输出同时用于：
  1. 结构分支（映射后加到 `s_inputs`，进入 pairformer/protenix）
  2. 对比分支（contrast loss）
- 新增 `esm_token_embedding` 预构建函数并在 forward 中统一写入
- 修正 `contrast_logit_scale` 传递，去掉前向中的提前 `exp`，避免与 loss 侧重复处理

### 4) 快速验证命令（先看 loss 是否正常刷新）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_API_KEY=... && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --project debug_qbiolip \
  --run_name ddp4_contrast_joint_quickcheck \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 50 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 0 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 5) 可直接训练（含对比学习）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_API_KEY=... && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --project debug_qbiolip \
  --run_name ddp4_contrast_joint_train \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 1000 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

备注：为避免 DDP 报 `logit_scale marked ready twice`，`contrast_logit_scale` 在前向输出时使用了 detached tensor。

## 稳定性改动（2026-03-03，contrast 联训）

### 6) 默认训练策略改为“联合训练”
- 模型配置 `protenix_mini_esm650m_unimol_contrast_v0.2.0` 已调整：
  - `contrast.train_projection_only: False`
  - `contrast.loss_weight: 0.2`
  - `contrast.loss_weight_warmup_steps: 2000`
- 目的：避免只训投影头导致结构分支几乎不下降，同时减轻 early-stage 多任务冲突。

### 7) 新增稳定版模型配置
- 新增：`protenix_mini_esm650m_unimol_contrast_v0.2.2`
- 关键默认：
  - `contrast.enable: True`
  - `contrast.train_projection_only: False`
  - `contrast.loss_weight: 0.2`
  - `contrast.loss_weight_warmup_steps: 2000`

### 8) loss 侧新增 contrast 权重 warmup
- 文件：`protenix/model/loss.py`
- 新增按 step 的动态权重：
  - `effective_weight = contrast.loss_weight * min(1, (step+1)/warmup_steps)`
- 并将实际权重写入日志指标：`contrast_loss/weight`。

### 9) 训练步数注入到 loss
- 文件：`runner/train.py`
- 在 `get_loss` 前向中注入：`input_feature_dict["current_step"] = self.step`
- 用于 loss 侧 warmup 权重计算。

### 10) 推荐训练命令（稳定版）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_API_KEY=... && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.2 \
  --project debug_qbiolip \
  --run_name ddp4_contrast_joint_train_v022 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 1000 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 11) 运行排障（W&B 无 loss 刷新）
- 现象：W&B run 已创建，但 `wandb-history.jsonl` 未生成，页面无 loss。
- 定位：训练在 mini-rollout 的链置换阶段反复报错并重启，错误样本见：
  - `output/ddp4_contrast_joint_train_v022_20260303_132037/errors/chain_permutation/T_20260303_132136.pt`
  - 核心报错：`The size of tensor a (0) must match the size of tensor b (235)`

### 12) 临时稳定启动参数（先让 loss 正常写入）
- 在命令里增加：
  - `--chain_permutation.train.mini_rollout False`
  - `--atom_permutation.train.mini_rollout False`
- 当前在跑：
  - `run_name: ddp4_contrast_joint_train_v022_noperm`
  - `model_name: protenix_mini_esm650m_unimol_contrast_v0.2.2`

### 13) contrast 缺失 key 统计与日志化
- 文件：`protenix/model/loss.py`
- 新增 `contrast` 输入字段缺失统计，不再只是静默 fallback。
- 记录维度：
  - `contrast_loss/missing_input_total`
  - `contrast_loss/missing_input_prot_z`
  - `contrast_loss/missing_input_lig_z`
  - `contrast_loss/missing_input_logit_scale`
  - `contrast_loss/missing_input_valid_mask`
- 机制：
  - 当以下任一字段缺失时注入稳定 fallback tensor，并将计数加一：
    - `contrast_prot_z`
    - `contrast_lig_z`
    - `contrast_logit_scale`
    - `contrast_valid_mask`
  - 将总缺失次数与各 key 缺失次数写入 WandB 统计。

### 14) 已明确为“一个卡一个样本”场景做适配
- 已按你要求按卡级输入处理（每 rank 处理独立样本）继续走训练流程，便于快速定位异常样本与重启场景。
