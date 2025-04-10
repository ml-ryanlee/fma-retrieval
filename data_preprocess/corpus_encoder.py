import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Import disable_caching along with load_dataset.
from datasets import load_dataset, disable_caching

# Disable caching so that the datasets won't write to disk.
disable_caching()

import soundfile as sf
import torch
import torchaudio
import numpy as np
import torchaudio.transforms as T
import random
from scipy import signal
import IPython.display as ipd
import json  # needed for jsonl output
from tqdm import tqdm  # optional, for progress visualization

from music2latent import EncoderDecoder
from music2latent.inference import UNet

import matplotlib.pyplot as plt

fma_dataset = load_dataset("ryanleeme17/free-music-archive-retrieval", split="train")

if torch.cuda.is_available():
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")


def get_models(self):
    gen = UNet().to(self.device)
    checkpoint = torch.load(self.load_path_inference, map_location=self.device, weights_only=False)
    gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
    self.gen = gen

EncoderDecoder.get_models = get_models

encdec = EncoderDecoder(device=torch_device)


def create_embeddings_jsonl(dataset, encdec, output_file):
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
    
    with open(output_file, "w") as fout:
        for sample in tqdm(dataset, desc="Encoding samples"):
            sample_output = {}
            # Loop over the audio columns to create embeddings.
            for key in embed_keys:
                audio_data = sample.get((key))
                audio_data = audio_data["array"]
                # audio_features = np.array(query_data["audio_features"])
                # latent = encdec.encode(audio_features)
                # example_sampling_rate = ds_example_0["audio"]["sampling_rate"]
                # example_audio = ds_example_0["audio"]["array"] 
                # print(query_data)
                # audio_features = np.array(query_data["audio_features"])
                # print(audio_features)
                if audio_data is None:
                    sample_output[key] = None
                
                else:
                    try:
                        # Assume encdec.encode() returns a tensor or list.
                        embedding = encdec.encode(audio_data)
                        # If embedding is a torch.Tensor, convert to a list.
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.detach().cpu().tolist()
                        sample_output[key] = embedding
                    except Exception as e:
                        print(f"Error encoding field '{key}' for sample with pid {sample.get('pid')}: {e}")
                        sample_output[key] = None
            # Append additional fields at the end.
            sample_output["pid"] = sample.get("pid", "")
            sample_output["title"] = sample.get("title", "")
            sample_output["genres"] = sample.get("genres", "")
            sample_output["qid"] = sample.get("qid", "")
            sample_output["q_audio_back_info"] = sample.get("q_audio_back_info", "")

            # Write the JSON object to a new line.
            fout.write(json.dumps(sample_output) + "\n")


if __name__ == '__main__':
    output_file = "embeddings.jsonl"
    create_embeddings_jsonl(fma_dataset, encdec, output_file)



