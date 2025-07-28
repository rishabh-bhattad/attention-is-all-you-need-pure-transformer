# This is an implementation of Attention is all you need Paper

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        # embed_size is how many dimensions a token is explained in. It is the dimensionality of input and outputs.
        # if each word/token is represented by a 512 dim vector. Embed_size will be 512.
        self.heads = heads
        # the number of parallel attention heads
        self.head_dim = embed_size // heads
        # Each head processes a smaller dimension of the embedding. If embed_size is 512, heads is 8,
        # then head_dim is 64. That is each head is only looking at 64 dim.

        assert (self.head_dim * heads == embed_size), "Embed size need to be div by heads"

        # Below is another way of doing projections for keys, values and queries

        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # For a token, this is the actual information a token is holding Value layer transformation the embeddings
        # into the meaning that the word contributes if it's deemed relevant by a query

        # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # For a token, keys shows what this word is offering, relevance to other words.
        # And keys layer allows each token to advertise its content and showcase its relevance to other words

        # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # For a token, query asks what information is important/relevant to me.
        # And query layer allows transformer to learn a specific way to "ask" for related information

        # GENERALLY, the values, keys and queries projections are linear layers of
        # nn.Linear(embed_size, embed_size) and then reshaped to (N, len, heads, head_dim)
        self.values_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)

        # To check similarity - we compare the query vector with key vector of every word (including itself)
        # This is done using dot product. If high dot product - it means high similarity.
        # This produces attention scores for each query, indicating its relevance to every other word

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        # It defines the final linear output layer for multi head attention

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # N represents Batch Size
        # Shape of values, keys and queries is [N, len, embed_size]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 2. Apply Linear Projections to the full embed_size
        # These layers transform the input token embeddings into Q, K, V representations
        # for attention. Each layer has its own set of learned weights.
        values = self.values_proj(values)  # Shape: (N, value_len, embed_size)
        keys = self.keys_proj(keys)  # Shape: (N, key_len, embed_size)
        queries = self.queries_proj(query)  # Shape: (N, query_len, embed_size)

        # 3. Split embedding into self.heads pieces AFTER projection
        # This is where the Multi-Head aspect becomes explicit. The embed_size
        # dimension is conceptually split into 'heads' number of smaller 'head_dim' chunks.
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # 4. Calculate Attention Energy (Dot Product of Queries and Keys)
        # energy = Q @ K_T for each head and each batch element
        # "nqhd, nkhd->nhqk" means:
        # n: batch size (aligned)
        # q: query length (outer product with k)
        # h: heads (aligned)
        # d: head_dim (dot product between these dimensions)
        # k: key length (outer product with q)
        # Resulting shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

        # Queries shape: (N, query_len, heads, heads_dim)
        # Keys shape: (N, key_len, heads, heads_dim)
        # Energy shape: (N, heads, query_len, key_len)

        # 5. Apply Mask (if provided)
        # This prevents attention to padding tokens or future tokens (in decoder)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 6. Scale and Apply Softmax to get Attention Probabilities
        # attention = softmax(energy / sqrt(d_k))
        attention = torch.softmax(energy / self.head_dim ** (1/2), dim=3)
        # The dim=3 means softmax is applied across the key_len dimension, so for each query
        # and each head, the sum of attention probabilities over all keys equals 1.
        # Resulting shape: (N, heads, query_len, key_len)

        # 7. Apply Attention Probabilities to Values (Weighted Sum)
        # out = Attention @ V
        # "nhql, nlhd->nqhd" means:
        # n: batch size (aligned)
        # h: heads (aligned)
        # q: query length (from attention)
        # l: key_len (aligned with value_len, for dot product)
        # d: head_dim (from values)
        # Resulting shape after einsum: (N, query_len, heads, head_dim)
        attention_output_per_head = torch.einsum("nhql, nlhd->nqhd", [attention, values])

        # 8. Concatenate Outputs from all Heads
        # The outputs from individual heads are now combined back into the original embed_size
        # for each token in the sequence.
        # This reshaping effectively concatenates the 'heads' dimension with the 'head_dim' dimension.
        out = attention_output_per_head.reshape(
            N, query_len, self.heads * self.head_dim  # This is equivalent to embed_size
        )

        # attention shape: (n, heads, query_len, key_len)
        # Values shape: (n, values_len, heads, heads_dim)
        # After einsum out -> (N, query_len, heads, heads_dim)

        # 9. Final Linear Projection (fc_out)
        # It takes the concatenated output from all heads and performs a final linear transformation.
        # This layer learns to combine and integrate the diverse information captured by each head
        # into a single, refined representation for each token.
        # Input shape: (N, query_len, heads * self.head_dim) which is (N, query_len, embed_size)
        # Output shape: (N, query_len, embed_size)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# In the original paper each head should have seperate weights, but in your code all heads share the same weights.
# here are two steps to fix it: 1. in __init__:  self.queries = nn.Linear(self.embed_size, self.embed_size,
# bias=False) (same for key and value weights) 2. in forward: put "queries = self.queries(queries)" above "queries =
# queries.reshape(...)" (also same for keys and values)


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask) # value, key and query are all the same that's why it is out, out, out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers =nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, trg_len)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # (trg_len, trg_len)
        trg_mask = trg_pad_mask & trg_sub_mask  # broadcasting (N, 1, trg_len, trg_len)
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 4, 3, 9, 5, 2, 0], [1, 8, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])

    print(out.shape)