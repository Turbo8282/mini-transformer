import torch
from model import Transformer
from data import src_vocab, tgt_vocab, encode
from utils import create_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Invert vocab: id → word
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

def greedy_decode(model, src_sentence, max_len=20):
    model.eval()
    
    src_ids = encode(src_sentence, src_vocab)
    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)  # (1, src_len)

    tgt_ids = [tgt_vocab["<sos>"]]
    tgt = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)  # (1, 1)

    for _ in range(max_len):
        src_mask, tgt_mask = create_mask(src, tgt, src_vocab["<pad>"], tgt_vocab["<pad>"])
        output = model(src, tgt, src_mask, tgt_mask)  # (1, tgt_len, vocab_size)
        next_token_logits = output[0, -1]  # (vocab_size,)
        next_token = torch.argmax(next_token_logits).item()

        tgt_ids.append(next_token)
        tgt = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)

        if next_token == tgt_vocab["<eos>"]:
            break

    words = [inv_tgt_vocab.get(idx, "<unk>") for idx in tgt_ids[1:]]
    return " ".join(words[:-1] if words[-1] == "<eos>" else words)

def load_trained_model():
    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab)
    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, embed_dim=128, num_heads=4, ff_dim=512, num_layers=2)
    model.load_state_dict(torch.load("mini_transformer.pt", map_location=DEVICE))
    return model.to(DEVICE)

if __name__ == "__main__":
    model = load_trained_model()
    sentence = input("Translate: ")
    result = greedy_decode(model, sentence)
    print("→", result)
