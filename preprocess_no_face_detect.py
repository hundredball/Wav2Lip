import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from os import listdir, path

import cv2
from tqdm import tqdm

from hparams import hparams as hp

parser = argparse.ArgumentParser()

parser.add_argument('--ncpu', help='Number of GPUs across which to run in parallel', default=8, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'


def process_video_file(vfile, args, cpu_id):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    video_stream = cv2.VideoCapture(vfile)

    i = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), frame)
        i = i+1


def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print('Started processing for {} with {} CPUs'.format(args.data_root, args.ncpu))

    filelist = glob(path.join(args.data_root, '*/*.mp4'))

    jobs = [(vfile, args, i % args.ncpu) for i, vfile in enumerate(filelist)]
    p = ProcessPoolExecutor(args.ncpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


if __name__ == '__main__':
    main(args)
