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

import laion_clap
from laion_clap.training.data import get_audio_features
from laion_clap.clap_module.linear_probe import LinearProbe
import librosa

fma_dataset = load_dataset("ryanleeme17/free-music-archive-retrieval", split="train")

if torch.cuda.is_available():
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")

def create_embeddings_jsonl(dataset, model: LinearProbe, output_file):
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
                    audio_data = librosa.resample(y=audio_data, orig_sr=44100, target_sr=48000)
                    audio_data = torch.from_numpy(audio_data).float().to(torch_device)  # Move to the same device as the model
                    audio_dict = {}
                    audio_dict = get_audio_features(
                        audio_dict, audio_data, 480000, 
                        data_truncating='rand_trunc', 
                        data_filling='repeatpad',
                        audio_cfg={},
                        require_grad=False
                    )
                    audio_dict["waveform"] = audio_dict["waveform"].reshape(1, -1)

                    # Pass the tensor to the model
                    embedding = model(audio_dict, device=torch_device).flatten()

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
    clap_module = laion_clap.CLAP_Module(enable_fusion=False)
    model = LinearProbe(model=clap_module.model, mlp=True, freeze=True, in_ch=512, out_ch=512, act='None')
    # model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=torch.device('cpu'))["model_state_dict"])
    model.load_state_dict(torch.load("checkpoints/best_model.pt")["model_state_dict"])
    model.to(torch_device)
    model.eval()
    output_file = "clap_embeddings.jsonl"
    create_embeddings_jsonl(fma_dataset, model, output_file)
    print(f"Time taken to create embeddings: {time.time() - start_time:.2f} seconds")
    