import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from data import get_dataloaders, SRC_PAD_IDX, TGT_PAD_IDX
from data import src_vocab, tgt_vocab
import pickle

from utils import create_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]   # Input to decoder
        tgt_output = tgt[:, 1:]  # Target for loss

        src_mask, tgt_mask = create_mask(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(logits, tgt_output)

        if torch.isnan(loss):
            print("💥 NaN loss detected")
            print("logits shape:", logits.shape)
            print("tgt_output shape:", tgt_output.shape)
            print("logits:", logits)
            print("tgt_output:", tgt_output)
            print("logits vocab size:", logits.size(-1))
            print("max tgt_output:", tgt_output.max().item())
            print("min tgt_output:", tgt_output.min().item())
            exit()

        # Backward + optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab)
    EMBED_DIM = 256
    NUM_HEADS = 8
    FF_DIM = 1024
    NUM_LAYERS = 4
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 5e-4

    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    train_loader = get_dataloaders(batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    torch.save(model.state_dict(), "mini_transformer.pt")
    with open("src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open("tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)
    print("✅ Model and vocab saved. You can now run inference!")

if __name__ == "__main__":
    main()
