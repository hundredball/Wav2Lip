import os

import torch
from torch.utils import data as data_utils

from Evaluator import eval_model
from inference_img import inference_rife
from inference import frame_inference, setup_wav2lip, set_model
from DatasetLowFrame import DatasetLowFrame
from models import Wav2Lip_disc_qual


def wav2lip_rife(mels, imgs, imgs_wrong):
    I0=imgs[:,:,0,:,:]
    I1=imgs[:,:,1,:,:]
    I2=imgs[:,:,2,:,:]
    rife_imgs0=inference_rife(I0,I1)
    rife_imgs1=inference_rife(I1,I2)
    out_frames=frame_inference(rife_imgs0+rife_imgs1[1::], mels, imgs_wrong)
    return out_frames

if __name__ == '__main__':
    
    setup_wav2lip()

    Wav2lip_weights="./checkpoints/wav2lip.pth"
    set_model(Wav2lip_weights)
    
    data_root = './lrs2_preprocessed/'
    dataset_low = DatasetLowFrame(data_root, 'test')
    test_data_loader = data_utils.DataLoader(dataset_low, batch_size=1, shuffle=False,num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_model(test_data_loader, wav2lip_rife, device, eval_steps=10)
