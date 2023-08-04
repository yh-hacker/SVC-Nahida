import math
import multiprocessing
import os
import argparse
from random import shuffle

import paddle
from glob import glob
from tqdm import tqdm

import utils
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import librosa
import numpy as np

hps = utils.get_hparams_from_file("configs/config.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length


def process_one(filename, hmodel):
    # print(filename)
    wav, sr = librosa.load(filename, sr=sampling_rate)
    soft_path = filename + ".soft.pdtensor"
    if not os.path.exists(soft_path):
        devive = "cuda" if paddle.device.is_compiled_with_cuda() else "cpu"
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = paddle.to_tensor(wav16k).cpu() if devive=='cpu' else paddle.to_tensor(wav16k).cuda()
        c:paddle.Tensor = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k)
        paddle.save(c.cpu(), soft_path)
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0 = utils.compute_f0_dio(wav, sampling_rate=sampling_rate, hop_length=hop_length)
        np.save(f0_path, f0)


def process_batch(filenames):
    print("正在加载内容的HuBERT……")
    device = "cuda" if paddle.device.is_compiled_with_cuda() else "cpu"
    hmodel = utils.get_hubert_model()
    print("HuBERT已被装载。")
    for filename in tqdm(filenames):
        process_one(filename, hmodel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="dataset/44k", help="path to input dir")

    args = parser.parse_args()
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method('spawn',force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [filenames[i:i + chunk_size] for i in range(0, len(filenames), chunk_size)]
    print([len(c) for c in chunks])
    processes = [multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks]
    for p in processes:
        p.start()
