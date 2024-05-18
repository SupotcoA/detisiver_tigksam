import os
import requests
import shutil

# 指定网址和目标文件夹
url = 'https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip'
target_folder = '/kaggle/working/ckpt'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 下载zip文件
response = requests.get(url)
response.raise_for_status()  # 确保请求成功

# 保存zip文件的临时路径
zip_file_path = os.path.join(target_folder, 'vq-f8-n256.zip')

# 将下载的内容写入临时zip文件
with open(zip_file_path, 'wb') as f:
    f.write(response.content)

# 使用shutil解压zip文件
shutil.unpack_archive(zip_file_path, target_folder)

try:
    shutil.rmtree('/kaggle/working/detisiver_tigksam')
except:
    pass

# ! git clone https://github.com/SupotcoA/detisiver_tigksam.git

import torch
import os
import sys

sys.path.append('/kaggle/working/detisiver_tigksam/ldm')
from build_model import build_model
from train import train

assert __name__ == "__main__"

torch.manual_seed(42)

config = {'ver': 'maskgit_afhq_0518_v01',
          'description': 'first try',
          'outcome_root': '/kaggle/working',
          }
config['outcome_path'] = os.path.join(config['outcome_root'], config['ver'])
config['log_path'] = os.path.join(config['outcome_path'], 'log.txt')

if not os.path.exists(config['outcome_path']):
    os.makedirs(config['outcome_path'])

maskgit_config = {'n_tokens': 256,
                  'n_pos': 1024,
                  'embed_dim': 256,
                  'num_heads': 8,
                  'fc_dim': 1024,
                  'n_layers': 8,
                  'n_steps': 10,
                  'c_dim': 4,
                  'n_classes': 3,
                  'dropout': 0.1}

ddconfig = {'double_z': False,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': (1, 2, 2, 4),
            'num_res_blocks': 2,
            'attn_resolutions': (32,),
            'dropout': 0.0}

ae_config = {'latent_size': (32, 32),
             'n_embed': 256,
             'embed_dim': 4,
             'ckpt_path': '/kaggle/working/ckpt/model.ckpt',
             'ddconfig': ddconfig
             }

train_config = {'train_steps': 30000,
                'log_path': config['log_path'],
                'log_every_n_steps': 5000,
                'eval_every_n_steps': 5000,
                'outcome_root': config['outcome_path'],
                'batch_size': 15,
                'base_learning_rate': 3.0e-4,
                'use_lr_scheduler': True,
                }

data_config = {'afhq_root': '/kaggle/input/afhq-512',
               'image_size': 256,
               'batch_size': train_config['batch_size'],
               'x_path': '/kaggle/input/afhq-kl-enc/x.pt',
               'cls_path': '/kaggle/input/afhq-kl-enc/cls.pt',
               'split': 0.94,
               'n_classes': 3,
               'log_path': config['log_path'],
               }

with open(config['log_path'], 'w') as f:
    for cf in [config, maskgit_config,
               ae_config, data_config, train_config]:
        f.write(str(cf) + '\n')

model, optim, lr_scheduler = build_model(maskgit_config,
                                         ae_config,
                                         train_config)
if not os.path.exists(data_config['x_path']):
    from data import build_dataset_img

    build_dataset_img(model, data_config)
else:
    from data import build_cached_dataset

    train_dataset, test_dataset = build_cached_dataset(data_config)
    train(model,
          optim,
          lr_scheduler,
          train_config,
          train_dataset,
          test_dataset)

shutil.make_archive(os.path.join(config['outcome_path']),
                    "zip", os.path.join(config['outcome_path']))
