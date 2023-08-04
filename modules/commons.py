import math
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

def slice_pitch_segments(x, ids_str, segment_size=4):
  ret = paddle.zeros_like(x[:, :segment_size])
  for i in range(x.shape[0]):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, idx_str:idx_end]
  return ret

def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
  b, d, t = x.shape
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (paddle.rand([b]) * ids_str_max.astype('float32')).astype(dtype='int64')
  ret = slice_segments(x, ids_str, segment_size)
  ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
  return ret, ret_pitch, ids_str

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
      m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = paddle.to_tensor([item for sublist in l for item in sublist],).flatten().astype('int32')
  return pad_shape


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
  """KL(P||Q)"""
  kl = (logs_q - logs_p) - 0.5
  kl += 0.5 * (paddle.exp(2. * logs_p) + ((m_p - m_q)**2)) * paddle.exp(-2. * logs_q)
  return kl


def rand_gumbel(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = paddle.rand(shape) * 0.99998 + 0.00001
  return -paddle.log(-paddle.log(uniform_samples))


def rand_gumbel_like(x):
  g = rand_gumbel(x.shape).astype(dtype=x.dtype)
  return g


def slice_segments(x, ids_str, segment_size=4):
  ret = paddle.zeros_like(x[:, :, :segment_size])
  for i in range(x.shape[0]):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (paddle.rand([b]) * ids_str_max).astype('int64')
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str


def rand_spec_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size
  ids_str = (paddle.rand([b]) * ids_str_max).astype('int64')
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str


def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = paddle.arange(length, dtype=np.float32)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (num_timescales - 1))
  inv_timescales = min_timescale * paddle.exp(
      paddle.arange(num_timescales, dtype=np.float32) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = paddle.concat([paddle.sin(scaled_time), paddle.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.reshape((1, channels, length))
  return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  b, channels, length = x.shape
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal.astype(dtype=x.dtype)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return paddle.concat([x, signal.astype(dtype=x.dtype)], axis)


def subsequent_mask(length):
  mask = paddle.tril(paddle.ones((length, length))).unsqueeze(0).unsqueeze(0)
  return mask


#@paddle.jit.to_static # @torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = paddle.tanh(in_act[:, :n_channels_int, :])
  s_act = paddle.nn.functional.sigmoid(in_act[:, n_channels_int:, :])
  print(t_act)
  print(s_act)
  acts = t_act * s_act
  return acts

def fix_pad_shape(pad_shape:paddle.Tensor, pad_tensor) -> paddle.Tensor: # 飞桨里面的padding函数对pad_shape有比较严格的要求，需要自己修正一下~~~
  if len(pad_tensor.shape) == 3:
    return pad_shape[0:2].astype('int32')
  elif len(pad_tensor.shape) == 4:
    return pad_shape[0:4].astype('int32')
  elif len(pad_tensor.shape) == 5:
    return pad_shape[0:6].astype('int32')
  return pad_shape.astype('int32')

def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x


def sequence_mask(length:paddle.Tensor, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = paddle.arange(max_length, dtype=length.dtype)
  return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  device = duration.device
  
  b, _, t_y, t_x = mask.shape
  cum_duration = paddle.cumsum(duration, -1)
  
  cum_duration_flat = cum_duration.reshape((b * t_x))
  path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
  path = path.reshape((b, t_x, t_y))
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose([0,1,3,2]) * mask
  return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, paddle.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = paddle.to_tensor(p.grad).norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      paddle.to_tensor(p.grad).clip_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm
