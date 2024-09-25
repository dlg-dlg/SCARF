import os, sys
import argparse
import shutil
import yaml
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.traincp import Evaluater
    
def test(subject_name, exp_cfg, args=None):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, exp_cfg)
    cfg = update_cfg(cfg, data_cfg)
    cfg.cfg_file = data_cfg
    cfg.group = data_type
    cfg.dataset.path = os.path.abspath(cfg.dataset.path)
    cfg.exp_name = f'{subject_name}_test'
    cfg.output_dir = os.path.join(args.exp_dir, data_type, subject_name, 'test')
    cfg.ckpt_path = os.path.abspath('/data1/local_userdata/dulingge/SCARF/exps/snapshot/male-3-casual/hybrid/model.tar') # any pretrained nerf model to have a better initialization
    
    os.makedirs(os.path.join(cfg.output_dir), exist_ok=True)
    # creat folders 
    os.makedirs(os.path.join(cfg.output_dir, 'logs'), exist_ok=True)
    # start training
    tester = Evaluater(config=cfg)

    tester.fit()

if __name__ == '__main__':
    from lib.utils.config import get_cfg_defaults, update_cfg
    parser = argparse.ArgumentParser()
    # parser.add_argument('--wandb_name', type=str, default = 'Test', help='project name')
    parser.add_argument('--exp_dir', type=str, default = './exps', help='exp dir')
    parser.add_argument('--data_cfg', type=str, default = 'configs/data/snapshot/male-3-casual.yml', help='data cfg file path')
    parser.add_argument('--exp_cfg', type=str, default = '/data1/local_userdata/dulingge/SCARF/configs/exp/stage_2_ft.yml', help='exp cfg file path')
    args = parser.parse_args()
    # 
    #-- data setting
    data_cfg = args.data_cfg
    data_type = data_cfg.split('/')[-2]
    subject_name = data_cfg.split('/')[-1].split('.')[0]
    
    #-- exp setting
    exp_cfg = args.exp_cfg
    # ### ------------- start training 
    test(subject_name, exp_cfg, args)
