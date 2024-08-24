import os
import torchaudio
from tqdm import tqdm
import numpy as np
from glob import glob
import torch

def extract_hubert(path):
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
    return embed


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

output_dir = "/fsx-project/xueyao/data/hubert_of_librilight_eval"
os.makedirs(output_dir, exist_ok=True)

for db in ["dev-clean"]:
    name2path = get_all_wav_paths(db)

    for filename, path in tqdm(name2path.items()):
        feat = extract_hubert(path)
        
        # print(filename)
        # print(feat.shape, feat)
        # exit()
        
        torch.save(feat.detach().cpu(), os.path.join(output_dir, f"{filename}.pt"))