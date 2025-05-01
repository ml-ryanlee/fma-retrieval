import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from datasets import load_dataset
import torch
import numpy as np
import json
from tqdm import tqdm
import time
import librosa

from music2latent.models import Encoder
from music2latent.audio import to_representation_encoder

# Load the FMA dataset
fma_dataset = load_dataset("ryanleeme17/free-music-archive-retrieval", split="train")

if torch.cuda.is_available():
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")

def create_embeddings_jsonl(dataset, model: Encoder, output_file):
    """
    For each sample in the dataset, compute embeddings for:
        - audio
        - q_audio
        - q_audio_eq
        - q_audio_pitch
        - q_audio_back
    using the music2latent encoder model.
    
    Then attach the extra fields 'pid', 'qid', and 'q_audio_back_info' to the resulting JSON object.
    Each line in the output file is a JSON dictionary mapping the (audio) embeddings and extra info.
    """
    # Define the keys we want to encode
    embed_keys = ["audio", "q_audio", "q_audio_eq", "q_audio_pitch", "q_audio_back"]
    model.eval()

    with open(output_file, "w") as fout:
        for sample in tqdm(dataset, desc="Encoding samples"):
            sample_output = {}
            # Loop over the audio columns to create embeddings
            for key in embed_keys:
                audio_data = sample.get(key)
                audio_data = audio_data["array"]
                if audio_data is None:
                    sample_output[key] = None
                else:
                    # Resample audio and convert to tensor
                    audio_data = librosa.resample(y=audio_data, orig_sr=44100, target_sr=48000)
                    audio_data = torch.from_numpy(audio_data).float().to(torch_device)
                    
                    # Convert to music2latent representation
                    audio_repr = to_representation_encoder(audio_data)
                    
                    # Pass through the model
                    with torch.no_grad():
                        embedding = model(audio_repr, extract_features=True)
                        embedding = embedding.flatten()  # Flatten the output
                    
                    # Convert embedding to list
                    embedding = embedding.detach().cpu().tolist()
                    sample_output[key] = embedding
            
            # Append additional fields
            sample_output["pid"] = sample.get("pid", "")
            sample_output["title"] = sample.get("title", "")
            sample_output["genres"] = sample.get("genres", "")
            sample_output["qid"] = sample.get("qid", "")
            sample_output["q_audio_back_info"] = sample.get("q_audio_back_info", "")

            # Write the JSON object to a new line
            fout.write(json.dumps(sample_output) + "\n")

if __name__ == '__main__':
    start_time = time.time()
    
    # Initialize the model
    model = Encoder()
    
    # Load the trained model weights
    checkpoint = torch.load("checkpoints/best_model.pt", map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device and set to eval mode
    model.to(torch_device)
    model.eval()
    
    # Create embeddings
    output_file = "m2l_embeddings.jsonl"
    create_embeddings_jsonl(fma_dataset, model, output_file)
    
    print(f"Time taken to create embeddings: {time.time() - start_time:.2f} seconds") 