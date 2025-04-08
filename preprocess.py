import torch

from torch.utils.data import Dataset


def tokenize(sentence):
    # Very basic tokenization: lowercase and split on whitespace
    return sentence.lower().replace("?", " ?").replace(".", " .").replace(",", " ,").split()

# Build vocab from a list of tokenized sentences
special_tokens = ["<pad>", "<sos>", "<eos>", "<UNK>"]

def build_vocab(tokenized_sentences, min_freq=1, special_tokens= special_tokens):
    vocab = {}
    word_freq = {}

    # Count word frequencies
    for sentence in tokenized_sentences:
        for token in sentence:
            word_freq[token] = word_freq.get(token, 0) + 1

    # Assign indices, ensuring special tokens are included
    index = 0
    for token in special_tokens:
        vocab[token] = index
        index += 1

    for token, freq in word_freq.items():
        if freq >= min_freq:
            vocab[token] = index
            index += 1

    return vocab

def numericalize(tokens, vocab, unk_token="<UNK>"):
    if unk_token not in vocab:
        raise ValueError(f"Unknown token '{unk_token}' must be in vocab")

    ans = [vocab.get(t, vocab[unk_token]) for t in tokens]  # Use <UNK> for OOV tokens

    max_token_id = max(ans) if ans else 0  # Handle empty cases safely

    # print(f"Max token index in dataset: {max_token_id}")
    # print(f"Vocabulary size: {len(vocab)}")

    return ans


def pad_sequences(sequences, pad_idx, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
        
    ans = [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]
    
    #print(f"Padding applied: {ans.shape}")
    # print(f"Max sequence length: {max_len}")
    
    return ans


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab, max_length):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_tokens = [self.src_vocab.get(word.lower(), 3) for word in src.split()]
        trg_tokens = [self.trg_vocab.get(word.lower(), 3) for word in trg.split()]
        
        # Add SOS/EOS and pad
        src_tokens = [1] + src_tokens + [2]
        trg_tokens = [1] + trg_tokens + [2]
        
        src_tokens += [0]*(self.max_length - len(src_tokens))
        trg_tokens += [0]*(self.max_length - len(trg_tokens))
        
        return (
            torch.tensor(src_tokens[:self.max_length], dtype=torch.long),
            torch.tensor(trg_tokens[:self.max_length], dtype=torch.long)
        )

