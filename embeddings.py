# import fasttext
from gensim.models import KeyedVectors
import torch
import os

def load_embeddings(lang):
    path = {
        'en': 'wiki-news-300d-1M.vec',
        'de': 'cc.de.300.vec',
        'ar': 'cc.ar.300.vec'
    }
    print(f"I will be importing the {lang} embeddings now / -> from embeddings.py file")
    embedding_path = os.path.join('D:\\projects\\Pre-trained embeddings', path[lang])
    # Since these are text files, set binary=False
    return KeyedVectors.load_word2vec_format(embedding_path, binary=False)

def create_embedding_matrix(vocab, embedding_model, embed_dim=300):
    matrix = torch.randn(len(vocab), embed_dim) * 0.1
    for word, idx in vocab.items():
        if word in embedding_model:
            matrix[idx] = torch.tensor(embedding_model[word])
    return matrix

