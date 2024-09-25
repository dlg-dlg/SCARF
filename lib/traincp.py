import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import wandb
from pytorch3d.loss import (
    mesh_edge_loss, 
)
from .models.scarf import SCARF
from .utils import util, rotation_converter, lossfunc
from .utils.config import cfg
from .datasets import build_datasets
import lpips
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
import cv2
import numpy as np
class Evametric(nn.Module):
    def __init__(self):
        super(Evametric, self).__init__()
        self.psnr = peak_signal_noise_ratio
        self.ssim = structural_similarity_index_measure
        self.lpips = lpips.LPIPS(net='alex')
    def forward(self, img, img_gt, mask=None):
        
        img_pnsr = self.psnr(img, img_gt, data_range=1.0)
        img_ssim = self.ssim(img, img_gt)
        img_lpips = self.lpips(img, img_gt)
        img_l1 = (img - img_gt).abs().mean()
        
        result = {
            'l1': img_l1,
            'psnr': img_pnsr,
            'ssim': img_ssim,
            'lpips': img_lpips.mean(),
        }
        return result
class PoseModel(nn.Module):
    def __init__(self, dataset, optimize_cam=False, use_perspective=False, 
                    use_deformation=False, deformation_dim=0):
        super(PoseModel, self).__init__()
        self.subject_id = dataset.subject_id    
        # first load data_param
        pose_dict = {}
        cam_dict = {}
        # exp_dict = {}
        for item in dataset:
            pose = item['full_pose']
            init_pose = rotation_converter.batch_matrix2axis(pose) + 1e-8
            cam = item['cam']
            # exp = item['exp'][:10]
            
            name = item['name']
            frame_id = item['frame_id']

            pose_dict[f'{name}_pose_{frame_id}'] = init_pose.clone()[None,...]
            cam_dict[f'{name}_cam_{frame_id}'] = cam.clone()[None,...]
            # exp_dict[f'{name}_exp_{frame_id}'] = exp.clone()[None,...]
        for key in pose_dict.keys():
            self.register_parameter(key, torch.nn.Parameter(pose_dict[key]))
        self.pose_dict = pose_dict

        for key in cam_dict.keys():
            self.register_parameter(key, torch.nn.Parameter(cam_dict[key]))
        self.cam_dict = cam_dict

        # for key in exp_dict.keys():
        #     self.register_parameter(key, torch.nn.Parameter(exp_dict[key]))
        # self.exp_dict = exp_dict
        self.use_perspective = use_perspective

    def forward(self, batch):
        # return poses of given frame_ids
        name = self.subject_id
        if 'cam_id' in batch.keys():
            cam_id = batch['cam_id']
            names = [f'{name}_{cam}' for cam in cam_id]
        else:
            names = [name]*len(batch['frame_id'])
        frame_ids = batch['frame_id']
        batch_size = len(frame_ids)
        batch_pose = torch.cat([getattr(self, f'{names[i]}_pose_{frame_ids[i]}') for i in range(batch_size)])
        batch_pose = rotation_converter.batch_axis2matrix(batch_pose.reshape(-1, 3)).reshape(batch_size, 55, 3, 3)
        batch['init_full_pose'] = batch['full_pose'].clone()
        batch['full_pose'] = batch_pose
        batch['full_pose'][:,22] = torch.eye(3).to(batch_pose.device)[None,...].expand(batch_size, -1, -1)
        
        batch['init_cam'] = batch['cam'].clone()
        batch['cam'] = torch.cat([getattr(self, f'{names[i]}_cam_{frame_ids[i]}') for i in range(batch_size)])

        # batch['exp'] = torch.cat([getattr(self,f'{names[i]}_exp_{frame_ids[i]}') for i in range(batch_size)])
        return batch
    
class Evaluater(torch.nn.Module):
    def __init__(self, config=None):
        super(Evaluater, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = cfg.device
        self.batch_size = self.cfg.train.batch_size
        self.image_size = self.cfg.dataset.image_size

        # load model
        self.prepare_data()
        init_beta = self.test_dataset[0]['beta']
        self.model = SCARF(self.cfg, device=self.device, init_beta=init_beta).to(self.device)
        self.posemodel = PoseModel(dataset=self.test_dataset, optimize_cam=self.cfg.opt_cam, use_perspective=self.cfg.use_perspective,
                                    use_deformation = False, deformation_dim=self.cfg.deformation_dim,
                                    ).to(self.device)
        self.configure_optimizers()
        self.load_checkpoint() 
        for k,param in self.model.named_parameters():
            param.requires_grad = False
        ### loss
        if self.cfg.loss.mesh_w_mrf > 0. or self.cfg.loss.w_patch_mrf:
            self.mrf_loss = lossfunc.IDMRFLoss().to(self.device)
        if self.cfg.loss.mesh_w_perceptual > 0. or self.cfg.loss.w_patch_perceptual:
            self.perceptual_loss = lossfunc.VGGPerceptualLoss().to(self.device)
            
        ### logger
        self.savefolder = self.cfg.output_dir
        logger.add(os.path.join(self.cfg.output_dir, 'logs', 'train.log'))
        logfolder = os.path.join(self.cfg.output_dir, 'logs')
        group = self.cfg.group
        self.evaluator = Evametric().to(self.device)
        # resume = None if self.cfg.clean else "allow"
        # wandb.init(
        #     id = f'{self.cfg.group}_{self.cfg.exp_name}',
        #     resume = resume,
        #     project=self.cfg.wandb_name, 
        #     name = self.cfg.exp_name,
        #     save_code = True,
        #     group = group,
        #     dir = logfolder)

    def configure_optimizers(self):
        parameters = []
        if self.cfg.opt_pose:
            parameters.append({'params': self.posemodel.parameters(), 'lr': self.cfg.train.pose_lr})
        self.optimizer = torch.optim.Adam(params=parameters)   
        self.decay_steps = [1000, 5000, 10000, 50000]; self.decay_gamma = 0.5
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay_steps, gamma=self.decay_gamma)
       
    def model_dict(self):
        current_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step
        }
        if self.cfg.opt_pose:
            current_dict['pose'] = self.posemodel.state_dict()
        return current_dict

    def load_checkpoint(self):
        self.global_step = 0
        model_dict = self.model_dict()
        # resume training, including model weight, opt, steps
        self.cfg.train.resume=False
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    if isinstance(checkpoint[key], dict):
                        util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
            
        elif os.path.exists(self.cfg.ckpt_path):
            checkpoint = torch.load(self.cfg.ckpt_path)
            key = 'model'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            key = 'pose'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            # if specify nerf ckpt, load and overwrite nerf weight
            if os.path.exists(self.cfg.nerf_ckpt_path):
                checkpoint = torch.load(self.cfg.nerf_ckpt_path)
                for param_name in model_dict['model'].keys():
                    if 'nerf' in param_name:
                        model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
            # if specify mesh ckpt, load and overwrite mesh weight
            if os.path.exists(self.cfg.mesh_ckpt_path):
                checkpoint = torch.load(self.cfg.mesh_ckpt_path)
                for param_name in model_dict['model'].keys():
                    if 'mlp_geo' in param_name or 'mlp_tex' in param_name or 'beta' in param_name:
                        if param_name in checkpoint['model']:
                            model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
            # if specify pose ckpt, load and overwrite pose weight
            if os.path.exists(self.cfg.pose_ckpt_path):
                checkpoint = torch.load(self.cfg.pose_path)
                util.copy_state_dict(model_dict['pose'], checkpoint['pose'])
        else:
            logger.info('model path not found, start training from scratch')

    def testoptim(self, batch, batch_nb, test=False):
        self.model.train()
        util.move_dict_to_device(batch, self.device)
        #-- update pose
        if not test:
            batch = self.posemodel(batch)
        #-- model: smplx parameters, rendering
        batch['global_step'] = self.global_step
        opdict = self.model(batch, train=True)
        #-- loss
        #### ----------------------- Losses
        losses = {}
        if self.cfg.use_nerf:
            ## regs for deformation
            if self.cfg.use_deformation and 'd_xyz' in batch.keys():
                losses['nerf_reg_dxyz'] = (batch['d_xyz']**2).sum(-1).mean()*self.cfg.loss.nerf_reg_dxyz_w
            gt_mask = batch['mask_sampled']
            if self.cfg.use_mesh:
                gt_mask = batch['cloth_mask_sampled']
            if self.cfg.loss.nerf_reg_normal_w > 0.:
                points_normal, points_neighbs_normal = self.model.canonical_normal(use_fine=False)
                losses['nerf_reg_normal'] = self.cfg.loss.nerf_reg_normal_w * F.mse_loss(points_normal, points_neighbs_normal)
                if self.cfg.use_fine and self.cfg.n_importance > 0 and not self.cfg.share_fine:
                    points_normal_fine, points_neighbs_normal_fine = self.model.canonical_normal(use_fine=True)
                    losses['nerf_reg_normal'] += self.cfg.loss.nerf_reg_normal_w * F.mse_loss(points_normal_fine, points_neighbs_normal_fine)
            losses['rgb'] = lossfunc.huber(opdict['rgbs']*batch['mask_sampled'], batch['image_sampled']*batch['mask_sampled'])*self.cfg.loss.w_rgb
            losses['alpha'] = lossfunc.huber(opdict['alphas'], gt_mask)*self.cfg.loss.w_alpha
            if self.cfg.use_fine and self.cfg.n_importance > 0 and not self.cfg.share_fine:
                losses['rgb_fine'] = lossfunc.huber(opdict['rgbs_fine']*batch['mask_sampled'], batch['image_sampled']*batch['mask_sampled'])*self.cfg.loss.w_rgb
                losses['alpha_fine'] = lossfunc.huber(opdict['alphas_fine'], gt_mask)*self.cfg.loss.w_alpha
                if self.cfg.sample_patch_rays:
                    patch_size = self.cfg.sample_patch_size
                    rgb_patch = opdict['rgbs_fine'][:,:patch_size**2] 
                    rgb_patch = opdict['rgbs_fine'][:,:patch_size**2] .reshape(-1, patch_size, patch_size, 3).permute(0,3,1,2)
                    rgb_patch_gt = batch['image_sampled'][:,:patch_size**2] .reshape(-1, patch_size, patch_size, 3).permute(0,3,1,2)
                    mask_patch = gt_mask[:,:patch_size**2].reshape(-1, patch_size, patch_size, 1).permute(0,3,1,2)
                    if self.cfg.loss.w_patch_mrf > 0.:
                        losses['nerf_patch_mrf'] = self.mrf_loss(rgb_patch*mask_patch, rgb_patch_gt*mask_patch)*self.cfg.loss.w_patch_mrf
                    if self.cfg.loss.w_patch_perceptual > 0.:
                        losses['nerf_patch_perceptual'] = self.perceptual_loss(rgb_patch*mask_patch, rgb_patch_gt*mask_patch)*self.cfg.loss.w_patch_perceptual
            
        if self.cfg.use_mesh and self.cfg.opt_mesh:
            if self.cfg.loss.geo_reg:
                mesh = opdict['mesh']
                offset = opdict['mesh_offset']
                new_verts = self.model.verts[None,...].expand(offset.shape[0], -1, -1) + offset
                batch_size = offset.shape[0] 
                losses["reg_offset"] = (opdict['offset']**2).sum(-1).mean()*self.cfg.loss.reg_offset_w
                exclude_idx = self.model.part_idx_dict['exclude']
                losses["reg_offset_fh"] = (opdict['offset'][:,exclude_idx]**2).sum(-1).mean()*self.cfg.loss.reg_offset_w_face
                losses["reg_offset_hand"] = (opdict['offset'][:,self.model.part_idx_dict['hand']]**2).sum(-1).mean()*self.cfg.loss.reg_offset_w_face*9.
                if self.cfg.loss.use_new_edge_loss:
                    offset = opdict['mesh_offset']
                    new_verts = self.model.verts[None,...].expand(offset.shape[0], -1, -1) + offset
                    losses["reg_edge"] = lossfunc.relative_edge_loss(new_verts, self.model.verts[None,...].expand(new_verts.shape[0], -1, -1), vertices_per_edge=self.model.verts_per_edge)*self.cfg.loss.reg_edge_w*100.
                else:
                    losses["reg_edge"] = mesh_edge_loss(mesh)*self.cfg.loss.reg_edge_w
                  
            if self.cfg.use_nerf:
                losses['mesh_skin_mask'] =lossfunc.huber(batch['skin_mask']*opdict['mesh_mask'], batch['skin_mask'])*self.cfg.loss.mesh_w_alpha_skin
                losses['mesh_inside_mask'] = (torch.relu(opdict['mesh_mask'] - batch['mask'])).abs().mean()*self.cfg.loss.mesh_inside_mask
                losses['mesh_image'] = lossfunc.huber(batch['skin_mask']*opdict['mesh_image'], batch['skin_mask']*batch['image'])*self.cfg.loss.mesh_w_rgb    
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.cfg.loss.mesh_w_alpha
                # skin color consistency
                tex = opdict['mesh_tex']
                if self.cfg.loss.skin_consistency_type == 'verts_all_mean':
                    losses['mesh_skin_consistency'] = (tex - tex.detach().mean(1)[:,None,:]).abs().mean()*self.cfg.loss.mesh_skin_consistency
                if self.cfg.loss.skin_consistency_type == 'verts_hand_mean':
                    all_idx = list(range(tex.shape[1]))
                    hand_idx = self.model.part_idx_dict['hand']
                    non_hand_idx = [i for i in all_idx if i not in self.model.part_idx_dict['exclude']]
                    losses['mesh_skin_consistency'] = (tex[:,non_hand_idx] - tex[:,hand_idx].detach().mean(1)[:,None,:]).abs().mean()*self.cfg.loss.mesh_skin_consistency
                if self.cfg.loss.skin_consistency_type == 'render_nonskin_mean':
                    losses['mesh_skin_consistency'] = (batch['cloth_mask']*(opdict['mesh_image'] - tex.detach().mean(1)[:,:,None,None]).abs()).mean()*self.cfg.loss.mesh_skin_consistency                
                if self.cfg.loss.skin_consistency_type == 'render_hand_mean':
                    hand_idx = self.model.part_idx_dict['hand']
                    losses['mesh_skin_consistency'] = (batch['cloth_mask']*(opdict['mesh_image'] - tex[:,hand_idx].detach().mean(1)[:,:,None,None]).abs()).mean()*self.cfg.loss.mesh_skin_consistency           
                if self.cfg.loss.mesh_w_mrf > 0.:
                    losses['mesh_image_mrf'] = self.mrf_loss(opdict['mesh_image']*batch['skin_mask'], batch['image']*batch['skin_mask'])*self.cfg.loss.mesh_w_mrf
                if self.cfg.loss.mesh_w_perceptual > 0.:
                    losses['mesh_image_perceptual'] = self.perceptual_loss(opdict['mesh_image']*batch['skin_mask'], batch['image']*batch['skin_mask'])*self.cfg.loss.mesh_w_perceptual
            else:
                losses['mesh_image'] = lossfunc.huber(opdict['mesh_image'], batch['image'])*self.cfg.loss.mesh_w_rgb
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.cfg.loss.mesh_w_alpha
                if self.cfg.loss.mesh_w_mrf > 0.:
                    losses['mesh_image_mrf'] = self.mrf_loss(opdict['mesh_image'], batch['image'])*self.cfg.loss.mesh_w_mrf
                if self.cfg.loss.mesh_w_perceptual > 0.:
                    losses['mesh_image_perceptual'] = self.perceptual_loss(opdict['mesh_image'], batch['image'])*self.cfg.loss.mesh_w_perceptual

        #########################################################d
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict

    def prepare_data(self):
        self.test_dataset = build_datasets.build_train(self.cfg.dataset, mode='test')
        logger.info('---- training data numbers: ', len(self.test_dataset))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)

    @torch.no_grad()
    def validation_step(self,data='testop', batch=None, returnVis=False):
        self.model.eval()
        batch = self.posemodel(batch)    
        batch['global_step'] = self.global_step
        opdict = self.model(batch, train=False)      
        frame_id = batch['frame_id'][0]
        savepath = os.path.join(self.cfg.output_dir, 'optimize', f'{frame_id}_{self.global_step:04}.jpg')
        fine_img = opdict['nerf_fine_image'][0]
        fine_img = fine_img.permute(1,2,0).cpu().numpy()
        fine_img = (fine_img*255).astype(np.uint8)
        fine_img = cv2.cvtColor(fine_img,cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepath,fine_img)
        logger.info(f'---- validation {data} step: {self.global_step}, save to {savepath}')
        return opdict

    def fit(self):
        for batch in tqdm(self.test_dataloader):
            self.global_step=0
            for step in tqdm(range(2000)):
                # if self.global_step % self.cfg.train.val_steps == 0:
                #     self.validation_step(data='testop')
                ## train
                losses, _ = self.testoptim(batch, self.global_step)
                all_loss = losses['all_loss']
                logger.info(f"{step}_{all_loss.item()}")
                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()

                self.global_step += 1
                self.scheduler.step()
                if (step+1)%200==0 or step==0:
                    opdict = self.validation_step(batch=batch)
                    pred = opdict['nerf_fine_image']
                    gt = batch['image'] 
                    val_metrics = self.evaluator(pred, gt)
                    val_info = f'step {step}'
                    for k, v in val_metrics.items():
                        val_info = val_info + f'{k}: {v:.6f}, '
                    logger.info(f"{step}_{val_info}")
        torch.save(self.posemodel.model_dict(), os.path.join(self.cfg.output_dir, 'model.tar'))
        print('training done')