import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer
)

class TabTransformerModel(nn.Module):
    """
    A sequence-to-sequence model that maps a CQT spectrogram (encoder input)
    to a guitar-tablature token sequence (decoder output).

    Pipeline
    --------
    1. CQT (n_bins=84, T) → Conv1d → d_model -dim features per frame
    2. TransformerEncoder over the audio frames (“memory”)
    3. Token IDs → Embedding + learnable positional encoding
    4. TransformerDecoder (causal mask) conditions on encoder memory
    5. Linear projection → vocab logits

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (PAD = 0 must be included).
    d_model : int
        Hidden size for embeddings and Transformer layers.
    nhead : int
        Number of attention heads.
    num_encoder_layers / num_decoder_layers : int
        Depth of the encoder / decoder stacks.
    dim_feedforward : int
        Hidden size of the position-wise feed-forward networks inside layers.
    dropout : float
        Drop-out used in Transformer sub-modules.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        # ------------- 1) Token embedding (decoder side) -------------
        # PAD token index = 0  (must match data preprocessing)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Learnable positional encodings (B, L, d_model) – fixed max len = 512
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))

        # ------------- 2) CQT → d_model feature extractor -------------
        # Treat 84-bin CQT like a “1-D image” (freq axis = channels).
        self.cqt_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=84,        # n_bins
                out_channels=d_model,  
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )
        self.cqt_norm = nn.LayerNorm(d_model)

        # ------------- 3) Transformer encoder (audio) -----------------
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # ------------- 4) Transformer decoder (tokens) ----------------
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.decoder_input_norm = nn.LayerNorm(d_model)

        # ------------- 5) Output projection --------------------------
        self.output_linear = nn.Linear(d_model, vocab_size)

    # ================================================================= #
    #                            Forward pass                            #
    # ================================================================= #
    def forward(
        self,
        cqt: torch.Tensor,                   # (B, n_bins=84, T)
        tokens: torch.Tensor,                # (B, L)  decoder input IDs
        tgt_key_padding_mask=None            # (B, L)  1 = PAD / ignore
    ):
        # ------------------------------------------------------------- #
        # 1. Pre-processing / normalisation of CQT                      #
        # ------------------------------------------------------------- #
        # Replace NaN / ±Inf to avoid exploding gradients
        cqt = torch.nan_to_num(cqt, nan=0.0, posinf=10.0, neginf=-10.0)
        cqt = torch.clamp(cqt, min=-10.0, max=10.0)  # log-scale guard

        # Channel-wise z-score per sample      shape: (B, 84, T)
        cqt = (cqt - cqt.mean(dim=(1, 2), keepdim=True)) / (
            cqt.std(dim=(1, 2), keepdim=True) + 1e-6
        )

        # ------------------------------------------------------------- #
        # 2. Encoder stack                                              #
        # ------------------------------------------------------------- #
        x = self.cqt_conv(cqt.transpose(1, 2))    # → (B, d_model, T)
        x = x.permute(0, 2, 1)                    # → (B, T, d_model)
        x = self.cqt_norm(x)

        memory = self.encoder(x)

        # ------------------------------------------------------------- #
        # 3. Token embedding + positional encoding                      #
        # ------------------------------------------------------------- #
        # Clamp to avoid out-of-range idx (safety for UNK etc.)
        tokens = torch.clamp(tokens, 0, self.embedding.num_embeddings - 1)
        x_embed = self.embedding(tokens)          # (B, L, d_model)

        # Add (learnable) position embeddings
        pos = self.positional_encoding[:, :x_embed.size(1), :]
        x_embed = x_embed + pos

        # Clean & normalise
        x_embed = torch.nan_to_num(x_embed, nan=0.0, posinf=10.0, neginf=-10.0)
        x_embed = self.decoder_input_norm(x_embed)

        # ------------------------------------------------------------- #
        # 4. Causal mask so each token sees ≤ previous tokens           #
        # ------------------------------------------------------------- #
        seq_len = x_embed.size(1)
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x_embed.device), diagonal=1
        ).bool()  # True above diagonal → masked / no attention

        # ------------------------------------------------------------- #
        # 5. Transformer decoder                                        #
        # ------------------------------------------------------------- #
        output = self.decoder(
            tgt=x_embed,                # (B, L, d_model)
            memory=memory,              # (B, T, d_model)
            tgt_mask=tgt_mask,          # (L, L)
            tgt_key_padding_mask=tgt_key_padding_mask  # (B, L)
        )
        output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)

        # ------------------------------------------------------------- #
        # 6. Final projection to vocab-size logits                      #
        # ------------------------------------------------------------- #
        return self.output_linear(output)          # (B, L, vocab_size)
