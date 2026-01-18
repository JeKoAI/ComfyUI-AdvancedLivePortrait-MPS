import os.path as osp
import numpy as np
import cv2
import torch
import yaml

from .utils.timer import Timer
from .utils.helper import concat_feat
from .utils.camera import headpose_pred_to_degree, get_rotation_matrix
from .config.inference_config import InferenceConfig
from .utils.rprint import rlog as log

# GLOBAL MPS setup
try:
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
except:
    DEVICE = torch.device("cpu")

def mps_fix_tensor(tensor):
    """Universal MPS Tensor Fix"""
    if tensor.device.type == 'mps':
        # 5D/6D -> 4D
        while tensor.dim() > 4 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.dim() == 6:  # [1,22,4,16,64,64] -> [1,16,64,64]
            tensor = tensor.mean(dim=[1,2], keepdim=False)
    return tensor

class LivePortraitWrapper(object):
    def __init__(self, cfg: InferenceConfig, appearance_feature_extractor, motion_extractor,
                 warping_module, spade_generator, stitching_retargeting_module):
        self.appearance_feature_extractor = appearance_feature_extractor.to(DEVICE)
        self.motion_extractor = motion_extractor.to(DEVICE)
        self.warping_module = warping_module.to(DEVICE)
        self.spade_generator = spade_generator.to(DEVICE)
        self.stitching_retargeting_module = stitching_retargeting_module
        
        # MPS: remove CUDA setting
        self.cfg = cfg
        self.device_id = DEVICE
        self.timer = Timer()
        
        # force FP32 (MPS does not supprt FP16 autocast)
        self.cfg.flag_use_half_precision = False

    def prepare_source(self, img: np.ndarray) -> torch.Tensor:
        """Prepare source image for MPS"""
        h, w = img.shape[:2]
        input_shape = getattr(self.cfg, 'input_shape', [256, 256])
        
        if h != input_shape[0] or w != input_shape[1]:
            x = cv2.resize(img, (input_shape[0], input_shape[1]))
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
            
        x = np.clip(x, 0, 1)
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        
        # MPS: .cuda() → .to(DEVICE)
        x = x.to(DEVICE)
        return mps_fix_tensor(x)

    def prepare_driving_videos(self, imgs) -> torch.Tensor:
        """Prepare driving videos for MPS"""
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        
        # MPS fix
        y = y.to(DEVICE)
        return mps_fix_tensor(y)

    def extract_feature_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 3D appearance features"""
        with torch.no_grad():
            # MPS: no CUDA autocast + FP16
            feature_3d = self.appearance_feature_extractor(mps_fix_tensor(x))
            return mps_fix_tensor(feature_3d.float())

    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
        """Extract keypoint information"""
        with torch.no_grad():
            # MPS: FP32 only
            kp_info = self.motion_extractor(mps_fix_tensor(x))

        flag_refine_info = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]
            kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]
            kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]
            kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)

        # MPS: fix all tensors
        for k, v in kp_info.items():
            if isinstance(v, torch.Tensor):
                kp_info[k] = mps_fix_tensor(v.float())
                
        return kp_info

    def transform_keypoint(self, kp_info: dict):
        """Transform keypoints with pose/exp"""
        kp = mps_fix_tensor(kp_info['kp'])
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']
        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        pitch = headpose_pred_to_degree(pitch)
        yaw = headpose_pred_to_degree(yaw)
        roll = headpose_pred_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3
        else:
            num_kp = kp.shape[1]

        rot_mat = get_rotation_matrix(pitch, yaw, roll)
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]

        return mps_fix_tensor(kp_transformed)

    def stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """Stitch keypoints"""
        feat_stiching = concat_feat(mps_fix_tensor(kp_source), mps_fix_tensor(kp_driving))
        with torch.no_grad():
            delta = self.stitching_retargeting_module['stitching'](feat_stiching)
        return mps_fix_tensor(delta)

    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """Apply stitching retargeting"""
        if self.stitching_retargeting_module is not None:
            bs, num_kp = kp_source.shape[:2]
            kp_driving_new = kp_driving.clone()
            delta = self.stitch(kp_source, kp_driving_new)

            delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)
            delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)

            kp_driving_new += delta_exp
            kp_driving_new[..., :2] += delta_tx_ty
            return mps_fix_tensor(kp_driving_new)

        return mps_fix_tensor(kp_driving)

    def warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> dict:
        """Warp and decode to final image"""
        feature_3d = mps_fix_tensor(feature_3d)
        kp_source = mps_fix_tensor(kp_source)
        kp_driving = mps_fix_tensor(kp_driving)
        
        with torch.no_grad():
            # MPS: only FP32
            ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
            ret_dct['out'] = self.spade_generator(feature=ret_dct['out'])

            # MPS: fix all tensors
            for k, v in ret_dct.items():
                if isinstance(v, torch.Tensor):
                    ret_dct[k] = mps_fix_tensor(v.float())

        return ret_dct

    def parse_output(self, out: torch.Tensor) -> np.ndarray:
        """Parse output to numpy image"""
        out = mps_fix_tensor(out)
        out = out.cpu().numpy().transpose([0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)
        return np.clip(out * 255, 0, 255).astype(np.uint8)
