import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
import einops


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=40):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )

        self.rope = RotaryPositionalEmbeddings(
            dim=self.d_k, max_seq_len=max_seq_len, base=50
        )

    def forward(self, x, mask=None):
        _, seq_len, _ = x.shape

        eye_matrix = torch.eye(
            seq_len,
            device=x.device,
            dtype=torch.bool,
        )
        diagonal_mask = torch.zeros(
            (seq_len, seq_len), device=x.device, dtype=torch.float32
        )
        diagonal_mask = diagonal_mask.masked_fill(eye_matrix, float("-inf"))

        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = einops.rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = einops.rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)

        q = self.rope(q)
        k = self.rope(k)

        q = einops.rearrange(q, "b s h d -> b h s d")
        k = einops.rearrange(k, "b s h d -> b h s d")
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=diagonal_mask,
            dropout_p=0.0 if not self.training else 0.1,
            is_causal=False,
        )
        out = einops.rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)
        return out


class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward,
        layer_norm_eps,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRoPE(
            d_model,
            num_heads,
            dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Pre-norm: normalize before self-attention
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class HangmanNet(nn.Module):
    """
    BERT-style encoder for hangman masked language modeling.
    Takes masked word sequences and predicts original characters.
    """

    __constants__ = ["device", "hidden_dim"]

    def __init__(
        self,
        vocab_size=27,  # a-z + '{'
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.3,
        layer_norm_eps=5e-5,
        device="cpu",
    ):
        super(HangmanNet, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.char_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithRoPE(
                    d_model=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    layer_norm_eps=layer_norm_eps,
                    dropout=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(hidden_dim, 26)

        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.char_embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_projection(x)

        return x

    def predict_numpy(self, x):
        # the caller should call torch.no_grad() and model.eval()

        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self(x)
        return F.softmax(logits, dim=-1).cpu().detach().numpy()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
