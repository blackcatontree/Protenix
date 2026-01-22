# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# model configs for inference and training,
# such as: protenix-base, protenix-mini, protenix-tiny, protenix-constraint.
# protenix_{model_size}_{features}_{version}
# model_size: base, mini, tiny
# features: default, constraint, esm, etc, if multiple split by "-"
# version: v{x}.{y}.{z}

"""
Currently, the model_name support the following models.

|           Model Name                |    ESM/MSA/Constraint    | Model Parameters(M ) |
|-------------------------------------|--------------------------|----------------------|
| `protenix_base_default_v0.5.0`      |      ❌ / ✅ / ❌         |         368.09       |
| `protenix_base_constraint_v0.5.0`   |      ❌ / ✅ / ✅         |         368.30       |
| `protenix_mini_esm_v0.5.0`          |      ✅ / ✅ / ❌         |         135.22       |
| `protenix_mini_ism_v0.5.0`          |      ✅ / ✅ / ❌         |         135.22       |
| `protenix_mini_default_v0.5.0`      |      ❌ / ✅ / ❌         |         134.06       |
| `protenix_tiny_default_v0.5.0`      |      ❌ / ✅ / ❌         |         109.50       |
"""
model_configs = {
    "protenix_base_default_v0.5.0": {
        "model": {"N_cycle": 10},
        "sample_diffusion": {
            "N_step": 200,
        },  # the default setting for base model
    },
    "protenix_base_constraint_v0.5.0": {
        "model": {
            "sample_diffusion": {
                "N_step": 200,
            },  # the default setting for constraint model
            "N_cycle": 10,
            "constraint_embedder": {
                "pocket_embedder": {
                    "enable": True,
                },
                "contact_embedder": {
                    "enable": True,
                },
                "substructure_embedder": {
                    "enable": True,
                },
                "contact_atom_embedder": {
                    "enable": True,
                },
            },
        },
        "data": {
            "weightedPDB_before2109_wopb_nometalc_0925": {
                "constraint": {
                    "enable": True,
                    "pocket": {
                        "prob": 0.2,
                        "max_distance_range": {
                            "PP": [4, 15],
                            "LP": [3, 10],
                        },
                    },
                    "contact": {
                        "prob": 0.1,
                    },
                    "substructure": {
                        "prob": 0.5,
                        "size": 1,
                        "coord_noise_scale": 1,
                    },
                    "contact_atom": {
                        "prob": 0.1,
                        "max_distance_range": {
                            "PP": [2, 12],
                            "PL": [2, 15],
                        },
                        "min_distance": -1,
                        "group": "complex",
                        "distance_type": "atom",
                        "feature_type": "continuous",
                    },
                },
            },
            "recentPDB_1536_sample384_0925": {
                "constraint": {
                    "enable": True,
                },
            },
            "posebusters_0925": {
                "constraint": {
                    "enable": True,
                },
            },
        },
        "load_strict": False,  # If finetuning from base model, model arch has been changed,
        # it should be False, for inference, it should be True.
        "finetune_params_with_substring": [
            "constraint_embedder.substructure_z_embedder",
            "constraint_embedder.pocket_z_embedder",
            "constraint_embedder.contact_z_embedder",
            "constraint_embedder.contact_atom_z_embedder",
        ],
    },
    "protenix_mini_default_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for mini model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
    },
    "protenix_tiny_default_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for tiny model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 8,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
    },
    "protenix_mini_esm_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for mini model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "enable": True,
            "model_name": "esm2-3b",
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_ism_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for mini model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "enable": True,
            "model_name": "esm2-3b-ism",
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_esm_online_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_esm_trainable_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "esm_trainable": True,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_baseline_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # "protenix_mini_esm_online_v0.1.0": {
    #     # "sample_diffusion": {
    #     #     "gamma0": 0,
    #     #     "step_scale_eta": 1.0,
    #     #     "N_step": 5,
    #     # },  # the default setting for mini model, we didn't change diffusion settings here
    #     "model": {
    #         "N_cycle": 4,
    #         "msa_module": {
    #             "n_blocks": 1,
    #         },
    #         "pairformer": {
    #             "n_blocks": 16,
    #         },
    #         "diffusion_module": {
    #             "atom_encoder": {
    #                 "n_blocks": 1,
    #             },
    #             "transformer": {
    #                 "n_blocks": 8,
    #             },
    #             "atom_decoder": {
    #                 "n_blocks": 1,
    #             },
    #         },
    #     },
    #     "esm": {
    #         "esm_model_online": True,
    #         "enable": True,
    #         "model_name": "esm2-3b",
    #         "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
    #     },
    #     "load_strict": False,  # For inference, it should be True.
    #     "use_msa": False,  # For efficiency, this model does not use MSA by default.
    # },
    "protenix_mini_esm_trainable_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "esm_trainable": True,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_esm_trainable_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "esm_trainable": True,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm 650M training
    "protenix_mini_esm_trainable_650m_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": True,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/',
            "truncation_seq_length": 1024, 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm 650M training
    "protenix_mini_esm_trainable_650m_v0.1.0_unimol": {
        # 扩散模型
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        # protenix主干网络结构配置
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        # ESM2语言模型相关配置
        "esm": {
            "esm_model_online": True,
            "enable": True, 
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": True,
            "embedding_dim": 1280,
            "local_model_path": '/home/dataset-local/tmp/zsl/Protenix/weight/',
            "truncation_seq_length": 1024, 
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/home/dataset-local/tmp/zsl/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/home/dataset-local/tmp/zsl/Protenix/weight/mol_pre_no_h_220816.pt",
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm 650M freeze
    "protenix_mini_esm_650m_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm unimol contrast base
    "protenix_mini_esm650m_unimol_contrast_v0.2.0":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True, 
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": True,
            "embedding_dim": 1280,
            "local_model_path": '/home/dataset-local/tmp/zsl/Protenix/weight/',
            "truncation_seq_length": 1024, 
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/home/dataset-local/tmp/zsl/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/home/dataset-local/tmp/zsl/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable":False
        },
        "contrast":{
            "enable":True,
            "contrast_dim": 256,
            "loss_weight": 1.0,
            "freeze_esm":True,
            "freeze_unimol":True
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
        
    }
}
