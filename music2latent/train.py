import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import random
import os
from tqdm import tqdm
from music2latent.models import UNet, Encoder, Decoder

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_embeddings, positive_embeddings, negative_embeddings=None):
        """
        Args:
            query_embeddings: Tensor of shape [batch_size, embedding_dim]
            positive_embeddings: Tensor of shape [batch_size, embedding_dim]
            negative_embeddings: Optional tensor of shape [batch_size, num_negatives, embedding_dim]
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(query_embeddings * positive_embeddings, dim=1) / self.temperature
        
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=2)
            # Compute similarity to all negatives
            neg_sim = torch.bmm(query_embeddings.unsqueeze(1), 
                              negative_embeddings.transpose(1, 2)).squeeze(1) / self.temperature
            
            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(len(query_embeddings), device=query_embeddings.device, dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
        else:
            # If no negatives provided, use other positives in batch as negatives
            sim_matrix = torch.matmul(query_embeddings, positive_embeddings.t()) / self.temperature
            labels = torch.arange(len(query_embeddings), device=query_embeddings.device)
            loss = F.cross_entropy(sim_matrix, labels)
            
        return loss

class AudioAugmentation:
    def __init__(self):
        self.augmentations = [
            self.pitch_shift,
            self.time_stretch,
            self.add_noise
        ]
        
    def __call__(self, audio):
        # Randomly select and apply augmentations
        aug = random.choice(self.augmentations)
        return aug(audio)
    
    def pitch_shift(self, audio):
        # Pitch shift by -2 to 2 semitones
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=44100, n_steps=n_steps)
    
    def time_stretch(self, audio):
        # Time stretch by 0.8x to 1.2x
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def add_noise(self, audio):
        # Add random noise at -20dB SNR
        noise = np.random.randn(len(audio))
        noise_db = -20
        audio_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        noise_watts = 10 ** ((audio_db - noise_db) / 10)
        noise = noise * np.sqrt(noise_watts)
        return audio + noise

class FMADataset(Dataset):
    def __init__(self, root_dir, transform=None, segment_length=5):
        self.root_dir = root_dir
        self.transform = transform
        self.segment_length = segment_length
        self.sample_rate = 44100
        
        # Load all audio files
        self.audio_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.mp3', '.wav')):
                    self.audio_files.append(os.path.join(root, file))
                    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_files[idx]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract random segment
        if len(audio) > self.segment_length * self.sample_rate:
            start = random.randint(0, len(audio) - self.segment_length * self.sample_rate)
            audio = audio[start:start + self.segment_length * self.sample_rate]
        else:
            audio = np.pad(audio, (0, self.segment_length * self.sample_rate - len(audio)))
            
        # Convert to spectrogram
        spectrogram = librosa.stft(audio, n_fft=2048, hop_length=512)
        spectrogram = np.abs(spectrogram)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Normalize
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        
        # Add channel dimension
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Apply augmentation for query
        if self.transform:
            augmented_audio = self.transform(audio)
            augmented_spectrogram = librosa.stft(augmented_audio, n_fft=2048, hop_length=512)
            augmented_spectrogram = np.abs(augmented_spectrogram)
            augmented_spectrogram = librosa.power_to_db(augmented_spectrogram, ref=np.max)
            augmented_spectrogram = (augmented_spectrogram - augmented_spectrogram.min()) / (augmented_spectrogram.max() - augmented_spectrogram.min())
            augmented_spectrogram = np.expand_dims(augmented_spectrogram, axis=0)
        else:
            augmented_spectrogram = spectrogram
            
        return {
            'original': torch.FloatTensor(spectrogram),
            'augmented': torch.FloatTensor(augmented_spectrogram)
        }

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Move data to device
        original = batch['original'].to(device)
        augmented = batch['augmented'].to(device)
        
        # Get embeddings
        with torch.no_grad():
            # Original embeddings - no gradient needed as this is our target
            orig_emb = model.encoder(original, extract_features=False)
            
        # Augmented embeddings - need gradient for training
        aug_emb = model.encoder(augmented, extract_features=False)
        
        # Compute loss
        loss = loss_fn(aug_emb, orig_emb)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            original = batch['original'].to(device)
            augmented = batch['augmented'].to(device)
            
            orig_emb = model.encoder(original, extract_features=False)
            aug_emb = model.encoder(augmented, extract_features=False)
            
            loss = loss_fn(aug_emb, orig_emb)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, checkpoint_path='checkpoints'):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_path, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(checkpoint_path, 'best_model.pt'))
            
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt'))

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = UNet().to(device)
    
    # Create dataset and dataloaders
    transform = AudioAugmentation()
    train_dataset = FMADataset('path/to/fma/train', transform=transform)
    val_dataset = FMADataset('path/to/fma/val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-4,
        device=device
    )

if __name__ == '__main__':
    main() 