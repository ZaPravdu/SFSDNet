from importlib import import_module
from datasets.dataset import P2RDataset
from datasets.utils import get_testset
from config import cfg
import model_assembler
import torch
import matplotlib.pyplot as plt
import numpy as np

class TrainConfig():
    def __init__(self):
        self.project_name = 'SFSDNet'
        # project_name='test'
        self.experiment_name = 'Pseudo-only_attention_dens_recon'
        self.max_epochs=10
        self.resume = False
        if self.resume:
            self.ckpt_path = f'/home/mscs/houminqiu2/SFSDNet/weight/VIC/{self.experiment_name}/{self.experiment_name}-latest.ckpt'
        else:
            self.ckpt_path = None

        self.data_mode = 'MovingDroneCrowd' 
        datasetting = import_module(f'datasets.setting.{self.data_mode}')
        self.cfg_data = datasetting.cfg_data
        self.device = 'cuda'
        self.dataset_path = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd'
        self.pseudo_dens_root = '/public/houminqiu2/pseudo_density_map'
        # self.error_root = 'SDNet_error_map'
        self.scene_path = '/home/mscs/houminqiu2/SFSDNet/MovingDroneCrowd/test.txt'
        self.model_path = '/home/mscs/houminqiu2/SFSDNet/sdnet.pth'
        self.cfg = cfg
        self.shuffle = False

        self.batch_size = 64
        self.lr = 0.0001
        self.weight_decay = 1e-6
        self.shuffle=True
        self.freeze_backbone = True
        self.freeze_feature_fuse = False
        self.freeze_head = True
        self.weight_path ='/home/mscs/houminqiu2/SFSDNet/sdnet.pth'
        self.dens_recon = True

def de_normalize(img):
    """
    
    """
    mean = torch.tensor([117/255., 110/255., 105/255.])[:,None,None]
    std = torch.tensor([67.10/255., 65.45/255., 66.23/255.])[:,None,None]
    img = (img * std + mean)
    return img

if __name__ == '__main__':
    train_cfg = TrainConfig()
    test_loader, _ = get_testset(train_cfg, P2RDataset)
    model = model_assembler.P2RModel(train_cfg).cuda()
    for i, data in enumerate(test_loader):
        w_img,s_img, targets = data
    
        pseudo, recon, dots_map = model.single_test(w_img.cuda(), targets)
        # if i == 1:
        break

    pseudo_global_dens = pseudo[0][0]
    recon_global_dens = recon[0][0]
    global_dots_map = dots_map[0][0]
    single_img = w_img[0]
    single_img = de_normalize(single_img)
    
    # 将tensor转换为numpy数组以便可视化
    single_img_np = single_img.cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, 3]
    pseudo_global_dens_np = pseudo_global_dens.cpu().detach().numpy().squeeze()  # [H, W]
    recon_global_dens_np = recon_global_dens.cpu().detach().numpy().squeeze()  # [H, W]
    global_dots_map_np = global_dots_map.cpu().detach().numpy().squeeze()  # [H, W]
    
    # 创建可视化图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 显示原图
    axes[0, 0].imshow(single_img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 显示重建前的密度图
    im1 = axes[0, 1].imshow(pseudo_global_dens_np.squeeze(), cmap='jet')
    axes[0, 1].set_title('Pseudo Density Map (Before Reconstruction)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 显示重建后的密度图
    im2 = axes[1, 0].imshow(recon_global_dens_np.squeeze(), cmap='jet')
    axes[1, 0].set_title('Reconstructed Density Map (After Reconstruction)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 显示dot map标注的点
    axes[1, 1].imshow(single_img_np)
    # 在有人头位置的地方绘制红色点
    y_coords, x_coords = np.where(global_dots_map_np==1 )  # 假设阈值为0.5
    axes[1, 1].scatter(x_coords, y_coords, c='red', s=1, alpha=0.8)
    axes[1, 1].set_title('Dot Map Overlay on Original Image')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/mscs/houminqiu2/SFSDNet/visualization_result.png', dpi=300, bbox_inches='tight')
    # plt.show()