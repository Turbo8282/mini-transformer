import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from data import get_dataloaders, SRC_PAD_IDX, TGT_PAD_IDX
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

        # Backward + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    SRC_VOCAB_SIZE = 100
    TGT_VOCAB_SIZE = 100
    EMBED_DIM = 128
    NUM_HEADS = 4
    FF_DIM = 512
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-3

    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    train_loader = get_dataloaders(batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    torch.save(model.state_dict(), "mini_transformer.pt")
    print("✅ Model saved as mini_transformer.pt")

if __name__ == "__main__":
    main()
