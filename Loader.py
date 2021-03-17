import os
import torch
from Evaluator import eval_model
from inference_img import inference_rife
from inference import frame_inference, setup_wav2lip, set_model
setup_wav2lip()

Wav2lip_weights=".../wav2lip.pth"
set_model(Wav2lip_weights)

def wav2lip_rife(mels, imgs):
    I0=imgs[:,:,0,:,:]
    I1=imgs[:,:,1,:,:]
    I2=imgs[:,:,2,:,:]
    rife_imgs0=inference_rife(I0,I1)
    rife_imgs1=inference_rife(I1,I2)
    out_frames=frame_inference(rife_imgs0+rife_imgs1[1::], mels)
    out_frames = torch.tensor(out_frames).permute((3,0,1,2))
    out_frames=out_frames.unsqueeze(0).float()
    return out_frames

eval_model(test_data_loader, wav2lip_rife, device, disc, eval_steps=300)