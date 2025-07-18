import torch
from torch.utils.data import DataLoader, Dataset

# Dummy English → French sentence pairs
data_pairs = [
    ("I love cats", "J’aime les chats"),
    ("You are smart", "Tu es intelligent"),
    ("We play games", "Nous jouons à des jeux"),
    ("They eat rice", "Ils mangent du riz"),
    ("He drinks water", "Il boit de l’eau")
]

# Basic word-level tokenizer
def build_vocab(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3
    for sent in sentences:
        for word in sent.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

src_vocab = build_vocab([src for src, _ in data_pairs])
tgt_vocab = build_vocab([tgt for _, tgt in data_pairs])

SRC_PAD_IDX = src_vocab["<pad>"]
TGT_PAD_IDX = tgt_vocab["<pad>"]

def encode(sentence, vocab):
    tokens = sentence.lower().split()
    ids = [vocab.get(token, 0) for token in tokens]
    return [vocab["<sos>"]] + ids + [vocab["<eos>"]]

class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = encode(src, src_vocab)
        tgt_ids = encode(tgt, tgt_vocab)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=SRC_PAD_IDX, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=TGT_PAD_IDX, batch_first=True)
    return src_padded, tgt_padded

def get_dataloaders(batch_size=2):
    dataset = TranslationDataset(data_pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
