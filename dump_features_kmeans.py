import os
import torchaudio
from tqdm import tqdm
import numpy as np
from glob import glob
import torch

def extract_kmeans(path, C, Cnorm):
    """
    C: [D, n_clusters]
    """
    wav, sr = torchaudio.load(path)
    assert sr == 16000

    with torch.no_grad():
        all_layer_embed, _ = hubert_model.extract_features(
            waveforms=wav.cuda(),
            num_layers=18,
        )  # returns all layer embeddings upto output_layer

    # [1, T, D] -> [T, D]
    embed = all_layer_embed[-1].squeeze(0)  # take last layer embedding

    # Norm
    embed = (embed - mean) / std

    distance = (
        embed.pow(2).sum(1, keepdim=True)
        - 2 * torch.matmul(embed, C)
        + Cnorm
    )

    # [T,]
    codebook_indices = distance.argmin(dim=1)
    # print(codebook_indices)

    # [T, D]
    return C.T[codebook_indices]


def get_all_wav_paths(db_name):
    data_dir = os.path.join("/data/home/xueyao/workspace/dataset/LibriSpeech/", db_name)

    name2path = dict()
    for spk in glob(os.path.join(data_dir, "*")):
        for chap in glob(os.path.join(data_dir, spk, "*")):
            for utt in glob(os.path.join(data_dir, spk, chap, "*.flac")):
                name2path[os.path.basename(utt).split('.')[0]] = os.path.join(data_dir, spk, chap, utt)
    
    print(f"#{db_name} = {len(name2path)}")
    return name2path


hubert_model = torchaudio.pipelines.HUBERT_LARGE.get_model()
hubert_model = hubert_model.cuda()
hubert_model.eval()

stat_file = "/fsx-project/xueyao/data/hubert_large_l18/train/mean_std.npz"
stat = np.load(stat_file)
mean, std = stat["mean"], stat["std"]
mean = torch.from_numpy(mean).cuda()
std = torch.from_numpy(std).cuda()

kmeans_dir = "/fsx-project/xueyao/ckpt/kmeans/hubert_large_l18"
output_root = "/fsx-project/xueyao/data/kmeans_of_librilight_eval"
# output_root = "/data/home/xueyao/workspace/libri-light/data"

for c in [32, 64, 128, 256, 512, 1024]:
    output_dir = os.path.join(output_root, f"c{c}")
    os.makedirs(output_dir, exist_ok=True)

    # [D, n_clusters]
    centroid = np.load(os.path.join(kmeans_dir, f"c{c}.npy")).T
    centroid = torch.from_numpy(centroid)
    centroid_norm = torch.pow(centroid, 2).sum(0, keepdim=True)

    centroid = centroid.cuda()
    centroid_norm = centroid_norm.cuda()

    for db in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        name2path = get_all_wav_paths(db)

        for filename, path in tqdm(name2path.items()):
            feat = extract_kmeans(path, centroid, centroid_norm)
            
            # print(filename)
            # print(feat.shape, feat)
            # exit()
            
            torch.save(feat.detach().cpu(), os.path.join(output_dir, f"{filename}.pt"))