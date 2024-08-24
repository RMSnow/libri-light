import os
import torchaudio
from tqdm import tqdm
import numpy as np
from glob import glob
import torch
import yaml
import sys
sys.path.append("/data/home/xueyao/workspace/RepCodec/repcodec")
from RepCodec import RepCodec

@torch.no_grad()
def get_quantized_repcodecs(feature, repcodec_model) -> torch.Tensor:
    """Quantize feature tensor using repcodec.

    Args:
        feature (torch.Tensor): Feature tensor to quantize [B, D, T]

    Returns:
        torch.Tensor: Quantized feature tensor, [B, T]
    """
    x = repcodec_model.encoder(feature)
    z = repcodec_model.projector(x)
    _, idx = repcodec_model.quantizer.codebook.forward_index(z.transpose(2, 1))
    # Fetch only the first codebook
    return idx.squeeze(0)


def extract_repcodec(path, repcodec_model):
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

    # [1, T, D]
    embed = all_layer_embed[-1]  # take last layer embedding

    # [1, T]
    repcodec_tokens = get_quantized_repcodecs(
        embed.transpose(-1, -2), repcodec_model
    )
    # print("repcodec_tokens:", repcodec_tokens.shape, repcodec_tokens)

    codebook = repcodec_model.quantizer.codebook
    codebook = codebook.cuda()
    codebook.codebook = codebook.codebook.cuda()

    # [1, T, D] -> [T, D]
    center_vectors = codebook.lookup(repcodec_tokens)
    # print("center_vetors", center_vectors.shape, center_vectors)

    center_vectors = center_vectors.squeeze(0)
    return center_vectors


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

repcodec_dir = "/fsx-project/xueyao/ckpt/repcodec/hubert_large_l18"
output_root = "/fsx-project/xueyao/data/repcodec_of_librilight_eval"

for c in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    output_dir = os.path.join(output_root, f"c{c}")
    os.makedirs(output_dir, exist_ok=True)

    config_file = os.path.join(repcodec_dir, f"c{c}", f"hubert_large_l18_c{c}.yaml")
    model_file = os.path.join(repcodec_dir, f"c{c}", f"hubert_large_l18_c{c}.pkl")

    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    repcodec_model = RepCodec(**conf)

    repcodec_model.load_state_dict(
        torch.load(model_file, map_location="cpu")["model"][
            "repcodec"
        ]
    )
    repcodec_model = repcodec_model.cuda()
    repcodec_model.quantizer.initial()
    repcodec_model.eval()

    # db_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    db_sets = ["dev-clean"]

    for db in db_sets:
        name2path = get_all_wav_paths(db)

        for filename, path in tqdm(name2path.items()):
            feat = extract_repcodec(path, repcodec_model)
            
            # print(filename)
            # print(feat.shape, feat)
            # exit()
            
            torch.save(feat.detach().cpu(), os.path.join(output_dir, f"{filename}.pt"))