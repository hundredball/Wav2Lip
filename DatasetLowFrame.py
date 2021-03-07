import os
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
import audio

import torch
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2
from hparams import hparams, get_image_list

syncnet_T = 5
syncnet_mel_step_size = 16

class DatasetLowFrame(object):
    def __init__(self, data_root, split):
        self.all_videos = get_image_list(data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = np.random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            # Only select even number frames 
            even_img_names = []
            for img_name in img_names:
                file_name = img_name.split('/')[-1]
                index_dot = file_name.index('.')
                if (int(file_name[:index_dot])%2==0):
                    even_img_names.append(img_name)
            
            img_name = np.random.choice(even_img_names)
            wrong_img_name = np.random.choice(even_img_names)
            while wrong_img_name == img_name:
                wrong_img_name = np.random.choice(even_img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            low_frame_window = window[:,[0,2,4],:,:]
            y = window.copy()
            # window[:, :, window.shape[2]//2:] = 0.   Remember to mask out lower part before feeding into Wav2Lip

            wrong_window = self.prepare_window(wrong_window)
            low_frame_wrong_window = wrong_window[:,[0,2,4],:,:]
            
            xT = torch.FloatTensor(low_frame_window)                    # Correct 3 frames [Channel (3), Frame (3), Height (96), Width (96)]
            xW = torch.FloatTensor(low_frame_wrong_window)              # Random 3 frames  [Channel, Frame, Height, Width]
            mel = torch.FloatTensor(mel.T).unsqueeze(0)                 # mel for y [1,80,16]
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)     # mel for individual images of y [Frame (5),1,80,16]
            y = torch.FloatTensor(y)                                    # Correct 5 frames [Channel, Frame (5), Height, Width]
            return xT, xW, indiv_mels, mel, y
        
if __name__ == '__main__':
    data_root = '../lrs2_preprocessed_1/'
    #data_root = '../teams/ECE285_WI21_A00/22/lrs2_preprocessed_1/'
    dataset_low = DatasetLowFrame(data_root, 'val')
    
    data_loader = data_utils.DataLoader(dataset_low, batch_size=1, shuffle=False,num_workers=0)
    for i, (xT, xW, indiv_mels, mel, gt) in enumerate(data_loader):

        print('Shape of xT: ', xT.shape)
        print('Shape of xW: ', xW.shape)
        print('Shape of indiv_mels: ', indiv_mels.shape)
        print('Shape of mel: ', mel.shape)
        print('Shape of gt: ', gt.shape)

        break