import torch
import pickle
from model import Transformer
from utils import create_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab from file (same used in training)
with open("src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)
with open("tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

# Invert vocab: id → word
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

def encode(sentence, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in sentence.lower().split()]


def greedy_decode(model, src_sentence, max_len=20):
    model.eval()

    src_ids = encode(src_sentence, src_vocab)
    print("🧾 Input tokens:", src_sentence)
    print("🔢 Encoded src_ids:", src_ids)

    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

    tgt_ids = [tgt_vocab["<sos>"]]
    tgt = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)

    for _ in range(max_len):
        src_mask, tgt_mask = create_mask(src, tgt, src_vocab["<pad>"], tgt_vocab["<pad>"])
        output = model(src, tgt, src_mask, tgt_mask)  # (1, tgt_len, vocab_size)
        next_token_logits = output[0, -1]  # (vocab_size,)
        next_token = torch.argmax(next_token_logits).item()

        print(f"🔮 Step {_}: predicted token ID = {next_token}, word = {inv_tgt_vocab.get(next_token, '<unk>')}")

        tgt_ids.append(next_token)
        tgt = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)

        if next_token == tgt_vocab["<eos>"]:
            print("🛑 Reached <eos>")
            break

    print("📜 Final tgt_ids:", tgt_ids)
    print("📘 Decoded words:", [inv_tgt_vocab.get(idx, "<unk>") for idx in tgt_ids])

    # Decode to words
    words = []
    for idx in tgt_ids[1:]:
        word = inv_tgt_vocab.get(idx, "<unk>")
        if word == "<eos>":
            break
        words.append(word)
    return " ".join(words)


def load_trained_model():
    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab)
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=4
    )
    model.load_state_dict(torch.load("mini_transformer.pt", map_location=DEVICE))
    return model.to(DEVICE)


if __name__ == "__main__":
    model = load_trained_model()
    sentence = input("Translate: ")
    print(f"🗣️ You entered: '{sentence}'")
    result = greedy_decode(model, sentence)
    print("→", result)
