import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import Encoder
from audio import to_representation_encoder
from contrastive_audio_dataset.contrastive_audio_dataset import ContrastiveAudioDataset
from datasets import load_dataset

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

def train_one_epoch(model: Encoder, train_loader: DataLoader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Move data to device
        original = batch['original'].to(device).float()
        transformed = batch['transformed'].to(device).float()
        
        # Convert to music2latent representation
        orig_repr = to_representation_encoder(original)
        trans_repr = to_representation_encoder(transformed)
        
        # Forward pass
        orig_emb = model(orig_repr, extract_features=True)
        trans_emb = model(trans_repr, extract_features=True)
        
        # Reshape embeddings for contrastive loss
        orig_emb = orig_emb.reshape(orig_emb.size(0), -1)
        trans_emb = trans_emb.reshape(trans_emb.size(0), -1)
        
        # Compute loss
        loss = loss_fn(trans_emb, orig_emb)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model: Encoder, val_loader: DataLoader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            original = batch['original'].to(device).float()
            transformed = batch['transformed'].to(device).float()
            
            # Convert to music2latent representation
            orig_repr = to_representation_encoder(original)
            trans_repr = to_representation_encoder(transformed)
            
            orig_emb = model(orig_repr, extract_features=True)
            trans_emb = model(trans_repr, extract_features=True)
            
            # Reshape embeddings for contrastive loss
            orig_emb = orig_emb.reshape(orig_emb.size(0), -1)
            trans_emb = trans_emb.reshape(trans_emb.size(0), -1)
            
            loss = loss_fn(trans_emb, orig_emb)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, checkpoint_path='checkpoints'):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_path, exist_ok=True)
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        # Start time
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Epoch time: {time.time() - start_time:.2f}s')
        
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
    
    return train_loss_history, val_loss_history

def plot_loss(train_loss_history, val_loss_history):
    """Plot training and validation loss and save the plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot.png')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Start time
    start_time = time.time()
    
    # Create model
    model = Encoder()
    model = model.to(device)

    # Load FMA dataset
    fma_dataset = load_dataset("ryanleeme17/free-music-archive-retrieval", split='train')

    # Split dataset into train, validation and test sets
    train_test_split=0.7
    random.seed(42)
    dataset_size = len(fma_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_size = int(dataset_size * train_test_split)
    train_indices = indices[:train_size]

    fma_dataset = fma_dataset.select(train_indices) # Excluding the test set
    fma_dataset = fma_dataset.train_test_split(test_size=0.2, seed=42) # Splitting into train and validation sets
    
    # Create dataset and dataloaders
    train_dataset = ContrastiveAudioDataset(fma_dataset['train'], sample_rate=48000, audio_length=10)
    val_dataset = ContrastiveAudioDataset(fma_dataset['test'], sample_rate=48000, audio_length=10)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Train model
    train_loss_history, val_loss_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=5e-4,
        device=device
    )
    plot_loss(train_loss_history, val_loss_history)
    print("Training complete. Loss plot saved as 'loss_plot.png'.")
    
    # End time
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f'Total time taken: {elapsed_time_str}')

if __name__ == '__main__':
    main()
