import os
from importlib import import_module
import numpy as np
from config import cfg

from datasets.utils import get_testset
from inference.engine import pseudo_error_inference, consistency_inference
from model.utils import get_model


# load model

# def save_npy(data, model_folder, scene, output_type, frame, error_type):
#     scene_name, scene_id = scene.split('/')
#     main_path = os.path.join(model_folder, scene_name, scene_id, output_type)
#     file_name = f'{frame}_{error_type}.npy'
#     if not os.path.exists(main_path):
#         os.makedirs(main_path)
#     save_path = os.path.join(main_path, file_name)
#     np.save(save_path, data)


class InferConfig():
    def __init__(self):
        self.data_mode = cfg.DATASET
        self.datasetting = import_module(f'datasets.setting.{self.data_mode}')
        self.cfg_data = self.datasetting.cfg_data
        self.device = 'cuda'
        self.dataset_path = 'MovingDroneCrowd'
        self.pseudo_dens_root = 'pseudo_density_map'
        self.error_root = 'SDNet_error_map'
        self.scene_path = './test.txt'
        self.model_path = './sdnet.pth'
        self.cfg = cfg
        self.shuffle = False
        # self.

def main():
    infer_cfg = InferConfig()

    # load data
    test_loader = get_testset(infer_cfg)

    model = get_model(infer_cfg)
    # inference
    consistency_inference(test_loader, infer_cfg, model)
    # pseudo_error_inference(test_loader, infer_cfg, model)

    # ae_path = 'weight/VIC/VGGAE/epoch=03-val_loss=0.7279.ckpt'
    # model = model_assembler.VGGAE.load_from_checkpoint(checkpoint_path=ae_path).cuda()
    # model.eval()

if __name__ == '__main__':
    main()

# calculate metrics for density map

# calculate metrics for reconstruction

# calculate correlation factor

