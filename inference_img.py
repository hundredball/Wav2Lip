import os
import cv2
import torch
import argparse
from torch.nn import functional as F
from RIFE.model.RIFE_HDv2 import Model
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
def inference_rife(img0, img1):
    
    model = Model()
    model.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log'), -1)
    model.eval()
    model.device()

    exp = 1
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)
    
    img_list = [img0, img1]
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp
    
    images=[]
    for i in range(len(img_list)):
        images.append((img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    '''
    
    mid = model.inference(img0, img1).unsqueeze(2)
    img0 = img0.unsqueeze(2)
    img1 = img1.unsqueeze(2)
    
    print('mid.shape: ', mid.shape)
    print('img0.shape: ', img0.shape)
    images = torch.stack([img0, mid, img1], dim=2)
    '''
    return images
