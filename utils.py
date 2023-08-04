import os
import glob
import re
import sys
import argparse
import logging
import json
import subprocess
import random

import visualdl
import librosa
import numpy as np
from scipy.io.wavfile import read
import paddle
import requests

from paddle.nn import functional as F
from modules.commons import sequence_mask
#from hubert import hubert_model
MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = float(1127 * np.log(1 + f0_min / 700))
f0_mel_max = float(1127 * np.log(1 + f0_max / 700))

def normalize_f0(f0, x_mask, uv, random_scale = True):
    # calculate means based on x_mask
    uv_sum = paddle.sum(uv, axis = 1, keepdim = True)
    uv_sum[uv_sum == 0] = 9999
    means = paddle.sum(f0[:, 0, :] * uv, axis = 1, keepdim = True) / uv_sum

    if random_scale:
        factor = paddle.zeros((f0.shape[0], 1)).uniform_(0.8, 1.2)
    else:
        factor = paddle.ones([f0.shape[0], 1])
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if paddle.isnan(f0_norm).any(): # 如果存在非数字
        print('utils.py:44行：存在非数字，退出。')
        exit(0)
    return f0_norm * x_mask


def plot_data_to_numpy(x, y):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data



def interpolate_f0(f0):
    '''
    对F0进行插值处理
    '''

    data = np.reshape(f0, (f0.size, 1))

    vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]

    return ip_data[:,0], vuv_vector[:,0]


def compute_f0_parselmouth(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    import parselmouth
    x = wav_numpy
    if p_len is None:
        p_len = x.shape[0]//hop_length
    else:
        assert abs(p_len-x.shape[0]//hop_length) < 4, "pad length error"
    time_step = hop_length / sampling_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
    return f0

def resize_f0(x, target_len):
    source = np.array(x)
    source[source<0.001] = np.nan
    target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res

def compute_f0_dio(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    import pyworld
    if p_len is None:
        p_len = wav_numpy.shape[0]//hop_length
    f0, t = pyworld.dio(
        wav_numpy.astype(np.double),
        fs=sampling_rate,
        f0_ceil=800,
        frame_period=1000 * hop_length / sampling_rate,
    )
    f0 = pyworld.stonemask(wav_numpy.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return resize_f0(f0, p_len)

def 导出的时候才能跑通的f0_to_coarse(f0):
  f0_mel_min = paddle.to_tensor(float(1127 * np.log(1 + f0_min / 700)))
  f0_mel_max = paddle.to_tensor(float(1127 * np.log(1 + f0_max / 700)))
  #is_paddle = isinstance(f0, paddle.Tensor)
  f0_mel = 1127 * (1 + f0 / 700).log()# if is_paddle else 1127 * np.log(1 + f0 / 700)
  greater:paddle.Tensor = f0_mel > 0
  a1 = paddle.masked_select(f0_mel,greater)
  a = paddle.subtract(a1 , f0_mel_min)
  b = paddle.to_tensor(f0_bin - 2,dtype = 'float32')
  c = paddle.subtract(f0_mel_max , f0_mel_min)
  left = paddle.to_tensor(a.astype('float32') * b.astype('float32') / c.astype('float32') + 1)
  right = f0_mel
  f0_mel = paddle.where(greater, left, right)

  less_equal = paddle.less_equal(f0_mel , paddle.to_tensor(1.))
  f0_mel = paddle.where(less_equal,paddle.to_tensor(1.),f0_mel) # float
  greater = paddle.greater_than(f0_mel , paddle.to_tensor(f0_bin - 1,dtype = 'float32'))
  f0_mel = paddle.where(greater,paddle.to_tensor(f0_bin - 1,dtype = 'float32'),f0_mel)
  f0_coarse = (f0_mel + 0.5).astype('int64') #if is_paddle else np.rint(f0_mel).astype(np.int) # 改了
  assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
  return f0_coarse

def f0_to_coarse(f0):
  is_paddle = isinstance(f0, paddle.Tensor)
  f0_mel = 1127 * (1 + f0 / 700).log() if is_paddle else 1127 * np.log(1 + f0 / 700)
  f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

  f0_mel[f0_mel <= 1] = 1
  f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
  f0_coarse = (f0_mel + 0.5).astype('int64') #if is_paddle else np.rint(f0_mel).astype(np.int) # 改了
  assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
  return f0_coarse
import os

def get_hubert_model():
  if os.getcwd() == "/home/aistudio":
    vec_path = f"{os.getcwd()}/build/hubert/hubert4.0.onnx"
  else:
    vec_path = f"{os.getcwd()}/hubert/hubert4.0.onnx"
  import onnxruntime as ort
  print("从{}加载模型".format(vec_path))
  model = ort.InferenceSession(vec_path,providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])
  return model

def get_hubert_content(hmodel, wav_16k_tensor) -> paddle.Tensor: # 传入的模型和声音数组
  feats = wav_16k_tensor
  if feats.dim() == 2:  # 双通道
    feats = feats.mean(-1)
  assert feats.dim() == 1, feats.dim()
  feats = feats.reshape((1, 1, -1)).numpy()
  outputs = hmodel.run(
        None,
        {"source": feats.astype(np.float32)},
    )[0]
  return paddle.to_tensor(outputs.transpose((0,2,1)))

def load_checkpoint(checkpoint_path, model, optimizer:paddle.optimizer.Optimizer=None, skip_optimizer:bool=False):
    # assert os.path.isfile(checkpoint_path)
    checkpoint_dict = paddle.load(checkpoint_path)
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    try:
        trainers = checkpoint_dict["trainers"]
    except:
        trainers = ['最初作者信息丢失']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.set_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k] # 改过的
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception as e:
            print(e)
            print("错误，%s 不在检查点里面" % k)
            logger.info("%s 不在检查点里面" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.set_state_dict(new_state_dict)
    else:
        model.set_state_dict(new_state_dict)
    logger.info("加载检查点 '{}' (迭代次数 {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration, trainers


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, trainers:list[str]):
  logger.info("保存模型和优化器状态位于迭代次数{} 到 {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  paddle.save(
      {'model': state_dict,
       'iteration': iteration,
       'optimizer': optimizer.state_dict(),
       'learning_rate': learning_rate,
       'trainers':trainers
       }, 
       checkpoint_path,)

def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
  """通过删除保存的检查点来释放空间

  参数:
  path_to_models    --  模型路径
  n_ckpts_to_keep   --  要保留的检查点数量，不包括G_0.pdparams和D_0.pdparams
  sort_by_time      --  True -> 按时间顺序删除检查点
                        False -> 按字典顺序删除检查点
  """
  ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
  name_key = (lambda _f: int(re.compile('._(\d+)\.pdparams').match(_f).group(1)))
  time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
  sort_key = time_key if sort_by_time else name_key
  x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pdparams')], key=sort_key)
  to_del = [os.path.join(path_to_models, fn) for fn in
            (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
  del_info = lambda fn: logger.info(f".. 通过删除模型 {fn} 来释放空间")
  del_routine = lambda x: [os.remove(x), del_info(x)]
  rs = [del_routine(fn) for fn in to_del]

def summarize(writer:visualdl.writer.writer.LogWriter, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(tag = k, value = v, step = global_step)
  for k, v in histograms.items():
    writer.add_histogram(tag = k, values = v, step = global_step)
  for k, v in images.items():
    writer.add_image(tag = k, img = v, step = global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(tag = k, audio_array = v.numpy(), step = global_step, sample_rate = audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pdparams"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return paddle.to_tensor(data.astype(np.float32), dtype = 'float32'), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(init:bool=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')

  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams

def get_hparams_no_args(parser):

  args = parser.parse_args()
  model_dir = args.path

  config_path = args.config_path
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{}不是git存储库，因此将忽略哈希值比较。".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git散列值不同。{}（已保存）!={}（当前）".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger

def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to: {save_path}")
    else:
        print(f"Failed to download the file from: {url}")

def repeat_expand_2d(content:paddle.Tensor, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = paddle.zeros([content.shape[0], target_len], dtype='float32').cpu() \
        if 'cpu' in str(content.place) \
        else paddle.zeros([content.shape[0], target_len], dtype='float32').cuda()
    temp = paddle.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target

# 含有所有前面存下来的输入超参
class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

