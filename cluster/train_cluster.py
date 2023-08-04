import os
from glob import glob
from pathlib import Path
import paddle
import logging
import argparse
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import time
import random

def train_cluster(in_dir, n_clusters, use_minibatch=True, verbose=False):

    logger.info(f"正在从{in_dir}加载特征")
    features = []
    nums = 0
    for path in tqdm.tqdm(in_dir.glob("*.soft.pdtensor")):
        path = str(path)
        features.append(paddle.load(path).squeeze(0).numpy().T)
        # print(features[-1].shape)
    features = np.concatenate(features, axis=0)
    print(nums, features.nbytes/ 1024**2, "MB , 形状：",features.shape, features.dtype)
    features = features.astype(np.float32)
    logger.info(f"聚类特征的形状：{features.shape}")
    t = time.time()
    if use_minibatch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters,verbose=verbose, batch_size=4096, max_iter=80).fit(features)
    else:
        kmeans = KMeans(n_clusters=n_clusters,verbose=verbose).fit(features)
    print(time.time()-t, "s")

    x = {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
    }
    print("结束")

    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default="./dataset/44k",
                        help='path of training data directory')
    parser.add_argument('--output', type=Path, default="logs/44k",
                        help='path of model output directory')

    args = parser.parse_args()

    checkpoint_dir = args.output
    dataset = args.dataset
    n_clusters = 10000

    ckpt = {}
    for spk in os.listdir(dataset):
        if os.path.isdir(dataset/spk):
            print(f"正在给{spk}训练kmeans中……")
            in_dir = dataset/spk
            x = train_cluster(in_dir, n_clusters, verbose=False)
            ckpt[spk] = x

    checkpoint_path = checkpoint_dir / f"kmeans_{n_clusters}.pdparams"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    paddle.save(
        ckpt,
        str(checkpoint_path),
    )


    # import cluster
    # for spk in tqdm.tqdm(os.listdir("dataset")):
    #     if os.path.isdir(f"dataset/{spk}"):
    #         print(f"start kmeans inference for {spk}...")
    #         for feature_path in tqdm.tqdm(glob(f"dataset/{spk}/*.discrete.npy", recursive=True)):
    #             mel_path = feature_path.replace(".discrete.npy",".mel.npy")
    #             mel_spectrogram = np.load(mel_path)
    #             feature_len = mel_spectrogram.shape[-1]
    #             c = np.load(feature_path)
    #             c = utils.tools.repeat_expand_2d(torch.FloatTensor(c), feature_len).numpy()
    #             feature = c.T
    #             feature_class = cluster.get_cluster_result(feature, spk)
    #             np.save(feature_path.replace(".discrete.npy", ".discrete_class.npy"), feature_class)

