import time
import os
import random
import numpy as np
import paddle

import modules.commons as commons
import utils
from modules.mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text

# import h5py


"""Multi speaker version"""


class TextAudioSpeakerLoader(paddle.io.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = hparams.spk

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pdtensor")
        if os.path.exists(spec_filename):
            spec = paddle.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = paddle.squeeze(spec, 0)
            paddle.save(spec, spec_filename)

        spk = filename.split("/")[-2]
        spk = paddle.to_tensor([self.spk_map[spk]],dtype = 'int64')

        f0 = np.load(filename + ".f0.npy")
        f0, uv = utils.interpolate_f0(f0)
        f0 = paddle.to_tensor(f0,dtype = 'float32')
        uv = paddle.to_tensor(uv,dtype = 'float32')

        c = paddle.load(filename+ ".soft.pdtensor")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])


        lmin = min(c.shape[-1], spec.shape[-1])
        assert abs(c.shape[-1] - spec.shape[-1]) < 3, (c.shape[-1], spec.shape[-1], f0.shape, filename)
        assert abs(audio_norm.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None
        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1]-800)
            end = start + 790
            spec, c, f0, uv = spec[:, start:end], c[:, start:end], f0[start:end], uv[start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]

        return c, f0, spec, audio_norm, spk, uv

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = \
        paddle.sort(
            paddle.to_tensor([x[0].shape[1] for x in batch],dtype = 'int64'),
            axis=0, descending=True),\
        paddle.argsort(
            paddle.to_tensor([x[0].shape[1] for x in batch],dtype = 'int64'),
            axis=0, descending=True)

        max_c_len = max([x[0].shape[1] for x in batch])
        max_wav_len = max([x[3].shape[1] for x in batch])

        lengths = paddle.zeros((len(batch),),dtype = 'int64')
        c_padded = paddle.to_tensor(np.random.rand(len(batch), batch[0][0].shape[0], max_c_len),dtype = 'float32')
        f0_padded =  paddle.to_tensor(np.random.rand(len(batch), max_c_len),dtype = 'float32')
        spec_padded =  paddle.to_tensor(np.random.rand(len(batch), batch[0][2].shape[0], max_c_len),dtype = 'float32')
        wav_padded =  paddle.to_tensor(np.random.rand(len(batch), 1, max_wav_len),dtype = 'float32')
        spkids = paddle.zeros((len(batch), 1),dtype = 'int64')
        uv_padded = paddle.to_tensor(np.random.rand(len(batch), max_c_len),dtype = 'float32')

        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.shape[1]] = c
            lengths[i] = c.shape[1]

            f0 = row[1]
            f0_padded[i, :f0.shape[0]] = f0

            spec = row[2]
            spec_padded[i, :, :spec.shape[1]] = spec

            wav = row[3]
            wav_padded[i, :, :wav.shape[1]] = wav
            spkids[i, 0] = row[4]

            uv = row[5]
            uv_padded[i, :uv.shape[0]] = uv

        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded
