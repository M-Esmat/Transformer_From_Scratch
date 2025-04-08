import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

import My_dataset as md
from preprocess import build_vocab, tokenize, numericalize, pad_sequences, TranslationDataset
import building_blocks as bb
import embeddings as emb
import tools

import os
from torch import save as torch_save
from torch import load as torch_load

print("Starting main.py file ...")

########## --- Dummy Transformer Check --- ##########
# #print(torch.cuda.is_available())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# x = torch. tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
# # 1 -> <sos>, 0 -> padding, 2 -> <eos>
# trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

# src_pad_idx = 0
# trg_pad_idx = 0
# src_vocab_size = 10
# trg_vocab_size = 10

# model = building_blocks.Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
# out = model(x, trg[:, :-1])
# print(out.shape)


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBED_DIM = 300
MAX_LENGTH = 20

print("I will be loading the embeddings ...")

# Pre-trained embeddings
en_emb = emb.load_embeddings('en')
de_emb = emb.load_embeddings('de')

# Tokenization and Vocabulary
print("will start tockenizing ...")

def build_vocab(sentences, min_freq=1):
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    word_counts = {}
    for sent in sentences:
        for word in sent.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    idx = 4
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab
print("will prepare dataset...")

# Build vocabularies
src_sentences = [pair[0] for pair in md.sampled_pairs]
trg_sentences = [pair[1] for pair in md.sampled_pairs]

src_vocab = build_vocab(src_sentences)
trg_vocab = build_vocab(trg_sentences)

# Create embedding matrices
src_embed_matrix = emb.create_embedding_matrix(src_vocab, en_emb)
trg_embed_matrix = emb.create_embedding_matrix(trg_vocab, de_emb)

print("will initialize the model...")

# Initialize model
model = bb.Transformer(
    src_vocab_size=len(src_vocab),
    trg_vocab_size=len(trg_vocab),
    src_pad_idx=0,
    trg_pad_idx=0,
    embed_size=EMBED_DIM,
    num_layers=2,
    heads=4,
    forward_expansion=4,
    dropout=0.1,
    device=DEVICE,
    max_length=MAX_LENGTH
)

# Replace embeddings with pre-trained
model.encoder.word_embedding.weight.data.copy_(src_embed_matrix)
model.decoder.word_embedding.weight.data.copy_(trg_embed_matrix)

BATCH_SIZE = 8
NUM_EPOCHS = 400
model = model.to(DEVICE)

# Dataset and DataLoader
dataset = TranslationDataset(md.sentence_pairs, src_vocab, trg_vocab, MAX_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


print("Starting the training loop ...")
# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)


print("Starting the training loop ...")

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for src, trg in loader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        
        # Compute loss
        loss = criterion(output.reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f'Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}')
    # Checkpointing
    # Save checkpoint every 50 epochs
    if (epoch+1) % 50 == 0:
        checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
        torch_save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'src_vocab': src_vocab,
            'trg_vocab': trg_vocab,
            'max_length': MAX_LENGTH
        }, checkpoint_path)
        #torch_save(checkpoint, f"checkpoints/model_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved for epoch {epoch+1} at {checkpoint_path}")

print("Starting Inference...")


test_model, src_vocab, trg_vocab = tools.load_test_model(
    checkpoint_path,
    TransformerClass=bb.Transformer,
    embed_size=EMBED_DIM,
    device=DEVICE,
    max_length=MAX_LENGTH)

print("testing translation ...")

# Test translation
test_sentence = "good morning"
print(f'Input: {test_sentence}')
print(f'Output: {tools.translate(test_sentence, test_model, src_vocab, trg_vocab, DEVICE)}')