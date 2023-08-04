import math
import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
import random
import paddle
import numpy as np
import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import soundfile as sf

def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)# than soundfile.
    except Exception as ex:
        print(f"¶ÁÈ¡'{full_path}'Ê§°Ü\nÒì³££º")
        print(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 32000
        else:
            raise Exception(ex)
    
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2# check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)
    
    if np.issubdtype(data.dtype, np.integer): # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    else: # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0) # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
    
    data = paddle.to_tensor(data.astype(np.float32),dtype = 'float32') / max_mag
    
    if (paddle.isinf(data) | paddle.isnan(data)).any() and return_empty_on_exception:# resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 32000
    if target_sr is not None and sampling_rate != target_sr:
        data = paddle.to_tensor(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return paddle.log(paddle.clip(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return paddle.exp(x) / C

class STFT():
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        
        self.n_mels     = n_mels
        self.n_fft      = n_fft
        self.win_size   = win_size
        self.hop_length = hop_length
        self.fmin     = fmin
        self.fmax     = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}
    
    def get_mel(self, y, center=False):
        sampling_rate = self.target_sr
        n_mels     = self.n_mels
        n_fft      = self.n_fft
        win_size   = self.win_size
        hop_length = self.hop_length
        fmin       = self.fmin
        fmax       = self.fmax
        clip_val   = self.clip_val
        
        if paddle.min(y) < -1.:
            print('min value is ', paddle.min(y))
        if paddle.max(y) > 1.:
            print('max value is ', paddle.max(y))
        
        if fmax not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[str(fmax)+'_'+str(y.device)] = paddle.to_tensor(mel,dtype = 'float32')
            self.hann_window[str(y.place)] = paddle.audio.functional.get_window('hann',self.win_size)
        
        y = paddle.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_length)/2), int((n_fft-hop_length)/2)), mode='reflect')
        y = y.squeeze(1)
        
        spec = paddle.signal.stft(y, n_fft, hop_length=hop_length, win_length=win_size, window=self.hann_window[str(y.device)],
                          center=center, pad_mode='reflect', normalized=False, onesided=True)
        # print(111,spec)
        spec = paddle.sqrt(spec.pow(2).sum(-1)+(1e-9))
        # print(222,spec)
        spec = paddle.matmul(self.mel_basis[str(fmax)+'_'+str(y.device)], spec)
        # print(333,spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        # print(444,spec)
        return spec
    
    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect

stft = STFT()
