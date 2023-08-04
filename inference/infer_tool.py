import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from inference import slicer

import librosa
import numpy as np
# import onnxruntime
import parselmouth
import soundfile
import paddle
import paddle.audio as paddleaudio
import paddleaudio

import cluster
#from hubert import hubert_model
import utils
from models import SynthesizerTrn,SynthesizerTrn_test

logging.getLogger('matplotlib').setLevel(logging.WARNING)
paddle.audio.backends.set_backend('soundfile')

def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()

def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])

def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr


class Svc(object):
    def __init__(self, net_g_path, config_path,
                 device=None,
                 cluster_model_path="./logs/44k/kmeans_10000.pdparams",mode="train"):
        self.net_g_path = net_g_path
        if device is None:
            self.dev = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
        else:
            self.dev = device
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.spk2id = self.hps_ms.spk
        # 加载hubert
        self.hubert_model = utils.get_hubert_model()
        self.load_model(mode)
        if os.path.exists(cluster_model_path):
            self.cluster_model = cluster.get_cluster_model(cluster_model_path)

    def load_model(self,mode):
        # 获取模型配置
        if mode == "train":
            self.net_g_ms = SynthesizerTrn(
                self.hps_ms.data.filter_length // 2 + 1,
                self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                **self.hps_ms.model)
        elif mode == "test":
            self.net_g_ms = SynthesizerTrn_test(
                self.hps_ms.data.filter_length // 2 + 1,
                self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                **self.hps_ms.model)
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        if "half" in self.net_g_path and paddle.device.is_compiled_with_cuda():
            self.net_g_ms.half().eval()
            self.net_g_ms.half().to(self.dev)
        else:
            self.net_g_ms.eval()
            self.net_g_ms.to(self.dev)



    def get_unit_f0(self, in_path, tran, cluster_infer_ratio, speaker):

        wav, sr = librosa.load(in_path, sr=self.target_sample)

        f0 = utils.compute_f0_parselmouth(wav, sampling_rate=self.target_sample, hop_length=self.hop_size)
        f0, uv = utils.interpolate_f0(f0)
        f0 = paddle.to_tensor(f0,dtype = ('float32'))
        uv = paddle.to_tensor(uv,dtype = ('float32'))
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        wav16k = librosa.resample(wav, orig_sr=self.target_sample, target_sr=16000)
        wav16k = paddle.to_tensor(wav16k)
        c = utils.get_hubert_content(self.hubert_model, wav_16k_tensor=wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        if cluster_infer_ratio !=0:
            cluster_c = cluster.get_cluster_center_result(self.cluster_model, c.cpu().numpy().T, speaker).T
            cluster_c = paddle.to_tensor(cluster_c,dtype = 'float32')
            c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv

    def infer(self, speaker, tran, raw_path,
              cluster_infer_ratio=0,
              auto_predict_f0=False,
              noice_scale=0.4):
        speaker_id = 0
        sid = paddle.to_tensor([int(speaker_id)],dtype = 'int64').unsqueeze(0)
        c, f0, uv = self.get_unit_f0(raw_path, tran, cluster_infer_ratio, speaker)
        if "half" in self.net_g_path and paddle.device.is_compiled_with_cuda():
            c = c.half()
        with paddle.no_grad():
            start = time.time()
            audio = self.net_g_ms.infer(c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0, noice_scale=noice_scale)[0,0].detach().astype('float32')
            use_time = time.time() - start
            print("vits耗时：{}".format(use_time))
        return audio, audio.shape[-1]

    def slice_inference(self,raw_audio_path, spk, tran, slice_db,cluster_infer_ratio, auto_predict_f0,noice_scale, pad_seconds=0.5,empty_cache=False):
        wav_path = raw_audio_path
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====分段开始，耗时{round(len(data) / audio_sr, 3)}秒======')
            # padd
            pad_len = int(audio_sr * pad_seconds)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            if slice_tag:
                print('跳过空段')
                _audio = np.zeros(length)
            else:
                out_audio, out_sr = self.infer(spk, tran, raw_path,
                                                    cluster_infer_ratio=cluster_infer_ratio,
                                                    auto_predict_f0=auto_predict_f0,
                                                    noice_scale=noice_scale
                                                    )
                _audio = out_audio.cpu().numpy()

            pad_len = int(self.target_sample * pad_seconds)
            _audio = _audio[pad_len:-pad_len]
            audio.extend(list(_audio))
        if empty_cache == True:
            paddle.device.cuda.empty_cache()
        return np.array(audio)


class RealTimeVC:
    def __init__(self):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000  # 区块长度
        self.pre_len = 3840  # 交叉淡化长度，640的倍数

    """输入输出都是1维numpy 音频波形数组"""

    def process(self, svc_model, speaker_id, f_pitch_change, input_wav_path):
        import maad
        audio, sr = paddleaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav)
            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len:2 * self.chunk_len]
