import glob
import os
import matplotlib
import paddle
from paddle.nn.utils import weight_norm
# matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight = paddle.normal(mean, std, m.weight.shape)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("加载'{}'".format(filepath))
    checkpoint_dict = paddle.load(filepath, map_location=device)
    print("完成。")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("保存检查点到{}".format(filepath))
    paddle.save(obj, filepath)
    print("完成。")


def del_old_checkpoints(cp_dir, prefix, n_models=2):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern) # get checkpoint paths
    cp_list = sorted(cp_list)# sort by iter
    if len(cp_list) > n_models: # if more than n_models models are found
        for cp in cp_list[:-n_models]:# delete the oldest models other than lastest n_models
            open(cp, 'w').close()# empty file contents
            os.unlink(cp)# delete file (move to trash when using Colab)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
