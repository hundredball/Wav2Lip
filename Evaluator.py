import os

from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual, FeatureExtractor
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

use_cuda = torch.cuda.is_available()

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

syncnet_T = 5
syncnet_mel_step_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
syncnet = SyncNet().to(device)

for p in syncnet.parameters():
    p.requires_grad = False
    
load_checkpoint('checkpoints/lipsync_expert.pth', syncnet, None, reset_optimizer=True, 
                                overwrite_global_states=False)

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss
    
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)
    
recon_loss = nn.L1Loss()
feature_extractor = FeatureExtractor()
feature_extractor.eval()

# --------- Add content loss here ---------------
def get_content_loss(g, gt):
    
    gen_feautres = feature_extractor(g)
    real_features = feature_extractor(gt)
    loss_content = recon_loss(gen_feautres, real_features.detach())

    return loss_content

def eval_model(test_data_loader, wav2lip_rife, device, eval_steps=300):
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss_RIFE, running_l1_loss_RIFE, running_content_loss_RIFE = [], [], []
    running_sync_loss_wav, running_l1_loss_wav, running_content_loss_wav = [], [], []

    for step, (x, xW, indiv_mels, mel, gt) in enumerate((test_data_loader)):

        x = x.to(device)
        xW = xW.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)

        # RIFE -> Wav2Lip
        g_RIFE, g_wav = wav2lip_rife(indiv_mels, x, xW, device)
        
        l1loss_RIFE = recon_loss(g_RIFE, gt)
        sync_loss_RIFE = get_sync_loss(mel, g_RIFE)
        content_loss_RIFE = get_content_loss(g_RIFE, gt)

        running_l1_loss_RIFE.append(l1loss_RIFE.item())
        running_sync_loss_RIFE.append(sync_loss_RIFE.item())
        running_content_loss_RIFE.append(content_loss_RIFE.item())
        
        l1loss_wav = recon_loss(g_wav, gt)
        sync_loss_wav = get_sync_loss(mel, g_wav)
        content_loss_wav = get_content_loss(g_wav, gt)

        running_l1_loss_wav.append(l1loss_wav.item())
        running_sync_loss_wav.append(sync_loss_wav.item())
        running_content_loss_wav.append(content_loss_wav.item())

        if step > eval_steps: break
    print('[Test] RIFE v.s. Target - L1: {}, Sync: {}, Content: {}'.format(step, sum(running_l1_loss_RIFE) / len(running_l1_loss_RIFE),
                                                        sum(running_sync_loss_RIFE) / len(running_sync_loss_RIFE),
                                                        sum(running_content_loss_RIFE) / len(running_content_loss_RIFE)))
    print('[Test] Wav2Lip v.s. Target - L1: {}, Sync: {}, Content: {}'.format(step, sum(running_l1_loss_wav) / len(running_l1_loss_wav),
                                                        sum(running_sync_loss_wav) / len(running_sync_loss_wav),
                                                        sum(running_content_loss_wav) / len(running_content_loss_wav)))
    
    save_sample_images(g_RIFE, g_wav, gt, './outputs/')
    
    return sum(running_sync_loss_wav) / len(running_sync_loss_wav)

def save_sample_images(g_RIFE, g_wav, gt, checkpoint_dir):
    g_RIFE = (g_RIFE.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g_wav = (g_wav.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    
    folder = checkpoint_dir
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((g_RIFE, g_wav, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])
