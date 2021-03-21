import os
import time
import argparse

import numpy as np
import cv2
import torch
from torch.utils import data as data_utils

from Evaluator import eval_model
from inference_img import inference_rife
from inference import frame_inference, setup_wav2lip, set_model
from DatasetLowFrame import DatasetLowFrame
from models import Wav2Lip_disc_qual

def wav2lip_rife(mels, imgs, imgs_wrong, device):
    
    out_frames_RIFE, out_frames_wav2lip = [], []
    for i in range(imgs.shape[0]):
        I0=imgs[i,:,0,:,:].unsqueeze(0)
        I1=imgs[i,:,1,:,:].unsqueeze(0)
        I2=imgs[i,:,2,:,:].unsqueeze(0)
        rife_imgs0=inference_rife(I0,I1)
        rife_imgs1=inference_rife(I1,I2)
        out_RIFE, out_wav2lip=frame_inference(rife_imgs0+rife_imgs1[1::], mels[i,:].unsqueeze(0), imgs_wrong[i,:].unsqueeze(0), device)
        out_frames_RIFE.append(out_RIFE.squeeze(0))
        out_frames_wav2lip.append(out_wav2lip.squeeze(0))
    
    out_frames_RIFE = torch.stack(out_frames_RIFE, dim=0)
    out_frames_wav2lip = torch.stack(out_frames_wav2lip, dim=0)
    return out_frames_RIFE, out_frames_wav2lip

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Inference code for RIFE+Wav2Lip')
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', default='./checkpoints/wav2lip.pth')
    parser.add_argument('--random_dataset', action = 'store_true', default=False)
    args = parser.parse_args()
    
    start_time = time.time()

    # Select Wav2Lip model
    Wav2lip_weights=args.checkpoint_path
    set_model(Wav2lip_weights)
    
    data_root = './lrs2_preprocessed/'
    dataset_low = DatasetLowFrame(data_root, 'test', args.random_dataset)
    test_data_loader = data_utils.DataLoader(dataset_low, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_model(test_data_loader, wav2lip_rife, device, eval_steps=1)
    print('Elapsed %.1f seconds'%(time.time()-start_time))
