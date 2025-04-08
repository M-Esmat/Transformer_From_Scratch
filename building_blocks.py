import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size= embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (self.head_dim * num_heads == embed_size), "embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.embed_size, self.embed_size, bias= False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias= False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias= False)
        self.fc_out = nn.Linear(num_heads*self.head_dim, embed_size)
        
    
    def forward(self, values, keys, query, mask):
        N= query.shape[0]
        
        # always correspond to source sentence lenght and target sentence lenght
        value_len, key_len, query_len= values.shape[1], keys.shape[1], query.shape[1]
        

        # First, apply linear projections on the original inputs.
        # Expected input shape: (N, seq_length, embed_size)
        queries = self.queries(query)   # shape: (N, query_len, embed_size)
        keys    = self.keys(keys)         # shape: (N, key_len, embed_size)
        values  = self.values(values)     # shape: (N, value_len, embed_size)
        
                           
        
        # split embedding into self.heads pieces
        values  = values.reshape(N, value_len, self.num_heads, self.head_dim) # self.num_heads and self.head_dim where one variable embed_size
        keys    = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)
        
 
        # the einsum method is more appropriate than using [ batch mat mul ( torch.bmm() )] after flatten n,num_heads
        # print("queries shape:", queries.shape)
        # print("keys shape:", keys.shape)

        out = torch.einsum('nqhd,nkhd->nhqk', queries, keys) # used in matrix multiplication for multiple dimensions
        
        # query shape: (N, query_len, num_heads, heads_dim) - nqhd
        # keys shape: (N, key_len, num_heads, heads_dim) - nkhd
        #output shape: (N, num_heads, query_len, key_len) - nhqk
        
        if mask is not None:
            # Mask shape: (N, 1, 1, key_len) or similar
            out = out.masked_fill(mask == 0, float('-1e20'))
            
        # apply attention function function - attention(q, k, v) = softmax(q*k.T / sqrt(dk)) * v
        attention = torch.softmax(out / math.sqrt(self.head_dim), dim=-1)  # dim= -1 === dim = 3
        
        # attention shape: (N, num_heads, query_len, key_len)
        # values shape: (N, value_len, num_heads, heads_dim)
        # output shape: (N, query_len, heads, head_dim)
        # print("attention shape:", attention.shape)
        # print("values shape:", values.shape)
        output = torch.einsum('nhqk,nkhd->nqhd', attention, values).reshape(N, query_len, self.num_heads * self.head_dim)
        
        output = self.fc_out(output)
        return output
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # transformer block = attention -> add&norm -> feed forward -> add&norm --> output [embeddings]
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)  # layer norm - norm for each example, batch norm - takes norm for each example
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),  # forward_expansion * embed
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
            
        )
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # apply the skip connection -> attention + query --> [multi-headattention -> add&norm]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        # apply second skip connection feed forward + x --> [ ff -> add&norm]
        out = self.dropout(self.norm2(forward + x))
        return out
    


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length=100):
        super(Encoder, self).__init__()
        self.embed_size= embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout= dropout, forward_expansion= forward_expansion)
                for _ in range(num_layers) #uncomment to use more than 1 block
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block= TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
        """
        Observation:
        standard Transformer decoders typically implement:

        Masked self-attention on the target.
        Separate cross-attention between the target and the encoder output.
        Your design combines cross-attention with a full TransformerBlock (which itself includes self-attention and feed-forward). 
        """
    def forward(self, x, value, key, src_mask, trg_mask):
        # Expand trg_mask to include num_heads dimension
        #trg_mask = trg_mask.unsqueeze(1)  # Shape becomes (N, 1, 1, trg_len, trg_len)
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(position)))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=128, num_layers=4, forward_expansion=2, heads=4, dropout=0, device='cuda', max_length=100):
        super(Transformer, self).__init__()
        self.name = "transformer_eng-german_trans"
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device,  forward_expansion, dropout, max_length)
        
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src): # if src pad idx then it will be set to 0 if not will be set to 1
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # Add num_heads dimension and ensure correct broadcasting
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)  # (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # print(f"Input tensor shape: {src.shape}")  # Expect [batch_size, seq_len]
        # print(f"input Mask shape: {src_mask.shape}")  # Expect [batch_size, seq_len]
        
        # print(f"trg tensor shape: {trg.shape}")  
        # print(f"target Mask shape: {trg_mask.shape}")  
        #print(f"Indexing tensor shape: {index_tensor.shape}")

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    
    
        