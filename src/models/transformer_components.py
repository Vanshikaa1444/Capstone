"""
Core Transformer Architecture Components for Sanskrit LLM
Includes multi-head attention, feed-forward networks, positional encoding, and normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence awareness in Sanskrit text"""
    
    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        # Calculate div_term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_length, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]
        Returns:
            Tensor with positional encoding added
        """
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for Sanskrit grammar understanding"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_length, d_model]
            key: [batch_size, seq_length, d_model]
            value: [batch_size, seq_length, d_model]
            mask: [batch_size, seq_length, seq_length] or None
        Returns:
            output: [batch_size, seq_length, d_model]
            attention_weights: [batch_size, num_heads, seq_length, seq_length]
        """
        batch_size, seq_length = query.size(0), query.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )
        
        output = self.w_o(context)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network with GELU activation for Sanskrit pattern modeling"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, d_model]
        Returns:
            output: [batch_size, seq_length, d_model]
        """
        # Apply first linear layer and GELU activation
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Apply second linear layer
        x = self.linear2(x)
        
        return x


class LayerNorm(nn.Module):
    """Layer normalization for training stability"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, d_model]
        Returns:
            normalized tensor
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer with Multi-Head Attention and FFN"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, d_model]
            mask: attention mask
        Returns:
            output: [batch_size, seq_length, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer with Masked Self-Attention and Cross-Attention"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: decoder input [batch_size, tgt_seq_length, d_model]
            encoder_output: encoder output [batch_size, src_seq_length, d_model]
            src_mask: source attention mask
            tgt_mask: target attention mask (causal mask)
        Returns:
            output: [batch_size, tgt_seq_length, d_model]
        """
        # Masked self-attention with residual connection and layer norm
        masked_attn_output, _ = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(masked_attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder with multiple layers"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, 
                 vocab_size: int, max_seq_length: int = 512, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_length]
            src_mask: attention mask
        Returns:
            output: [batch_size, seq_length, d_model]
        """
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return x


class TransformerDecoder(nn.Module):
    """Complete Transformer Decoder with multiple layers"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 vocab_size: int, max_seq_length: int = 512, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, tgt_seq_length]
            encoder_output: [batch_size, src_seq_length, d_model]
            src_mask: source attention mask
            tgt_mask: target attention mask (causal mask)
        Returns:
            output: [batch_size, tgt_seq_length, d_model]
        """
        # Embedding and positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        return x


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create padding mask for attention"""
    return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Create causal (look-ahead) mask for decoder"""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask == 0


def create_combined_mask(tgt: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create combined padding and causal mask for decoder"""
    seq_length = tgt.size(1)
    device = tgt.device
    
    # Padding mask
    pad_mask = create_padding_mask(tgt, pad_token_id)
    
    # Causal mask
    causal_mask = create_causal_mask(seq_length, device)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Combine masks
    combined_mask = pad_mask & causal_mask
    return combined_mask 