import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Import disable_caching along with load_dataset.
from datasets import load_dataset, disable_caching

# Disable caching so that the datasets won't write to disk.
# disable_caching()

import torch
import numpy as np
import json  # needed for jsonl output
from tqdm import tqdm  # optional, for progress visualization
import time
import music2latent as m2l
import torch.nn as nn
import torch.nn.functional as F
from laion_clap.training.data import get_audio_features
from laion_clap.clap_module.linear_probe import LinearProbe
import librosa
from torch.amp import autocast

fma_dataset = load_dataset("ryanleeme17/free-music-archive-retrieval", split="train")

if torch.cuda.is_available():
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")

def create_embeddings_jsonl(dataset, model: nn.LSTM, m2l_model: m2l.EncoderDecoder, output_file):
    """
    For each sample in the dataset, compute embeddings for:
        - audio
        - q_audio
        - q_audio_eq
        - q_audio_pitch
        - q_audio_back
    using the `encdec.encode()` method.
    
    Then attach the extra fields 'pid', 'qid', and 'q_audio_back_info' to the resulting JSON object.
    Each line in the output file is a JSON dictionary mapping the (audio) embeddings and extra info.
    """
    # Define the keys we want to encode.
    embed_keys = ["audio", "q_audio", "q_audio_eq", "q_audio_pitch", "q_audio_back"]
    model.eval()

    with open(output_file, "w") as fout:
        for sample in tqdm(dataset, desc="Encoding samples"):
            sample_output = {}
            # Loop over the audio columns to create embeddings.
            for key in embed_keys:
                audio_data = sample.get((key))
                audio_data = audio_data["array"]
                if audio_data is None:
                    sample_output[key] = None

                else:
                    # Resample audio and convert to tensor
                    audio_data = torch.from_numpy(audio_data).float().to(torch_device)  # Move to the same device as the model
                    audio_data = audio_data.reshape(1, -1)

                    with autocast("cuda"):
                        latent = m2l_model.encode(audio_data)

                        # Reshape and permute
                        latent = latent.permute(0, 2, 1)

                        output, (hn, cn) = model(latent)
                        embedding = torch.mean(output, dim=1).flatten()
                        embedding = F.normalize(embedding, p=2, dim=0)

                    # Convert embedding to list if necessary
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.detach().cpu().tolist()
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    sample_output[key] = embedding
            # Append additional fields at the end.
            sample_output["pid"] = sample.get("pid", "")
            sample_output["title"] = sample.get("title", "")
            sample_output["genres"] = sample.get("genres", "")
            sample_output["qid"] = sample.get("qid", "")
            sample_output["q_audio_back_info"] = sample.get("q_audio_back_info", "")

            # Write the JSON object to a new line.
            fout.write(json.dumps(sample_output) + "\n")


if __name__ == '__main__':
    start_time = time.time()
    m2l_model = m2l.EncoderDecoder(device=torch_device)
    model = nn.LSTM(64, 256, 2, batch_first=True, dropout=0.1, bidirectional=True, device=torch_device)
    # model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=torch.device('cpu'))["model_state_dict"])
    model.load_state_dict(torch.load("checkpoints/best_model.pt")["model_state_dict"])
    # model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_50.pt")["model_state_dict"])
    model.to(torch_device)
    model.eval()
    output_file = "clap_embeddings.jsonl"
    create_embeddings_jsonl(fma_dataset, model, m2l_model, output_file)
    print(f"Time taken to create embeddings: {time.time() - start_time:.2f} seconds")
    