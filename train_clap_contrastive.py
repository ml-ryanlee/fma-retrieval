import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import laion_clap
import librosa
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing
import time
from pathlib import Path
import numpy.core.multiarray

# macOS specific optimizations
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Reduce memory usage
os.environ['OMP_NUM_THREADS'] = str(
    multiprocessing.cpu_count())  # Optimize CPU usage

# Set cache directory
CACHE_DIR = Path.home() / ".cache" / "fma_dataset"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set device - use MPS (Metal Performance Shaders) if available on macOS
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Add safe globals for numpy
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])


class ContrastiveAudioDataset(Dataset):
    def __init__(self, dataset, transform_types=["q_audio", "q_audio_eq", "q_audio_pitch", "q_audio_back"], target_length=48000):
        self.dataset = dataset
        self.transform_types = transform_types
        # Target length in samples (1 second at 48kHz)
        self.target_length = target_length
        # Convert to list to get length
        self.data = list(dataset)
        # Pre-compute valid indices to avoid processing invalid samples
        self.valid_indices = []
        for idx in range(len(self.data)):
            try:
                sample = self.data[idx]
                if all(key in sample and sample[key] is not None for key in ["audio"] + transform_types):
                    self.valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

    def __len__(self):
        return len(self.valid_indices)

    def _process_audio(self, audio_data):
        """Process audio data to ensure consistent length"""
        # Resample to 48kHz
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=44100,
            target_sr=48000,
            res_type='kaiser_fast'
        )

        # Pad or truncate to target length
        if len(audio_data) > self.target_length:
            # Truncate to target length
            audio_data = audio_data[:self.target_length]
        elif len(audio_data) < self.target_length:
            # Pad with zeros to target length
            pad_length = self.target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_length), mode='constant')

        # Convert to float32
        return audio_data.astype(np.float32)

    def __getitem__(self, idx):
        try:
            sample = self.data[self.valid_indices[idx]]

            # Process original audio
            audio_data = self._process_audio(sample["audio"]["array"])

            # Get two random transformations for positive pair
            transform1, transform2 = np.random.choice(
                self.transform_types, 2, replace=False)

            # Process transformed audio
            audio1 = self._process_audio(sample[transform1]["array"])
            audio2 = self._process_audio(sample[transform2]["array"])

            return {
                "original": audio_data,
                "transform1": audio1,
                "transform2": audio2
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a valid sample with zeros if there's an error
            return {
                "original": np.zeros(self.target_length, dtype=np.float32),
                "transform1": np.zeros(self.target_length, dtype=np.float32),
                "transform2": np.zeros(self.target_length, dtype=np.float32)
            }


def info_nce_loss(embeddings1, embeddings2, temperature=0.07):
    """
    Compute InfoNCE loss for contrastive learning
    embeddings1: [batch_size, embedding_dim]
    embeddings2: [batch_size, embedding_dim]
    """
    batch_size = embeddings1.shape[0]

    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)

    # Create labels identifying positive pairs (diagonal is positive)
    labels = torch.arange(batch_size, device=device)

    # Compute similarity matrix for all possible pairs
    logits = torch.matmul(embeddings1, embeddings2.T) / temperature

    # Compute CrossEntropyLoss with positive pairs as targets
    # This automatically handles the numerator/denominator calculation properly
    loss = F.cross_entropy(logits, labels)

    return loss


def train_contrastive(model, train_loader, num_epochs, learning_rate=1e-4, checkpoint_dir=None):
    """Train model with contrastive learning and save checkpoints"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # Get numpy arrays from batch and ensure they're in the correct format
            transform1_np = batch["transform1"].numpy() if isinstance(
                batch["transform1"], torch.Tensor) else batch["transform1"]
            transform2_np = batch["transform2"].numpy() if isinstance(
                batch["transform2"], torch.Tensor) else batch["transform2"]

            # Ensure arrays are float32
            transform1_np = transform1_np.astype(np.float32)
            transform2_np = transform2_np.astype(np.float32)

            # Get embeddings for both transformations using numpy arrays
            with torch.no_grad():  # No gradients needed for feature extraction
                embeddings1_np = model.get_audio_embedding_from_data(
                    x=transform1_np)
                embeddings2_np = model.get_audio_embedding_from_data(
                    x=transform2_np)

            # Convert embeddings to tensors and move to device
            embeddings1 = torch.from_numpy(embeddings1_np).to(
                device).requires_grad_(True)
            embeddings2 = torch.from_numpy(embeddings2_np).to(
                device).requires_grad_(True)

            # Clear memory
            del transform1_np, transform2_np, embeddings1_np, embeddings2_np
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()

            # Compute loss
            loss = info_nce_loss(embeddings1, embeddings2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(
                {"loss": total_loss / (progress_bar.n + 1)})

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss}")

        # Save checkpoint after each epoch if checkpoint_dir is specified
        if checkpoint_dir:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"clap_checkpoint_epoch_{epoch+1}.pth")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_dir:
                best_model_path = os.path.join(
                    checkpoint_dir, "clap_best_model.pth")
                try:
                    torch.save(model.state_dict(), best_model_path)
                    print(
                        f"Best model saved to {best_model_path} with loss: {best_loss}")
                except Exception as e:
                    print(f"Error saving best model: {e}")

    return model


def load_dataset_with_retry(dataset_name, split, max_retries=3, retry_delay=5):
    """Load dataset with retry logic"""
    for attempt in range(max_retries):
        try:
            print(
                f"Attempting to load dataset (attempt {attempt + 1}/{max_retries})...")
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(CACHE_DIR),
                streaming=False  # Disable streaming for more reliable downloads
            )
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise


def load_clap_model():
    """Load CLAP model with proper checkpoint handling"""
    model = laion_clap.CLAP_Module(enable_fusion=False)

    # Load checkpoint with weights_only=False for trusted checkpoints
    try:
        # First try with weights_only=True
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            model.load_ckpt()
    except Exception as e:
        print("First attempt failed, trying with weights_only=False...")
        # If that fails, try with weights_only=False (only for trusted checkpoints)
        # Create a temporary function to override torch.load
        original_load = torch.load

        def custom_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        # Temporarily override torch.load
        torch.load = custom_load
        try:
            model.load_ckpt()
        finally:
            # Restore original torch.load
            torch.load = original_load

    return model


def main():
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Set model directory for saving
    model_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved to: {model_dir}")

    # Load dataset with retry logic
    try:
        fma_dataset = load_dataset_with_retry(
            "ryanleeme17/free-music-archive-retrieval",
            split="train[:1000]"  # Use only first 1000 samples for testing
        )
    except Exception as e:
        print(f"Failed to load dataset after multiple attempts: {e}")
        return

    # Create dataset and dataloader with optimized settings
    dataset = ContrastiveAudioDataset(fma_dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=8,  # Reduced batch size for macOS
        shuffle=True,
        num_workers=min(4, multiprocessing.cpu_count()),
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize and load CLAP model
    model = load_clap_model()
    model = model.to(device)
    model.train()  # Ensure model is in training mode

    # Train model
    model = train_contrastive(
        model, train_loader, num_epochs=10, checkpoint_dir=model_dir)

    # Save final trained model
    final_model_path = os.path.join(model_dir, "clap_contrastive_trained.pth")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved successfully to: {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")
        # Try an alternative save method
        try:
            torch.save(model, os.path.join(
                model_dir, "clap_contrastive_trained_full.pth"))
            print(f"Full model saved as alternative")
        except Exception as e2:
            print(f"Alternative save also failed: {e2}")


if __name__ == "__main__":
    main()
