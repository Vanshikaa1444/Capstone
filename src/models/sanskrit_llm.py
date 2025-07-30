"""
Main Sanskrit LLM Model
Combines encoder, decoder, and output processing for Question-Answering functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

from .transformer_components import (
    TransformerEncoder, TransformerDecoder, 
    create_padding_mask, create_combined_mask
)

class SanskritLLM(nn.Module):
    """
    Complete Sanskrit Large Language Model for Question-Answering
    Architecture: 8-layer transformer with 8 attention heads per layer
    """
    
    def __init__(self, config):
        super(SanskritLLM, self).__init__()
        
        self.config = config
        self.vocab_size = config.VOCAB_SIZE
        self.d_model = config.HIDDEN_SIZE
        self.num_layers = config.NUM_LAYERS
        self.num_heads = config.NUM_ATTENTION_HEADS
        self.d_ff = config.INTERMEDIATE_SIZE
        self.max_seq_length = config.MAX_SEQ_LENGTH
        self.dropout_rate = config.DROPOUT_RATE
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            vocab_size=self.vocab_size,
            max_seq_length=self.max_seq_length,
            dropout=self.dropout_rate
        )
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            vocab_size=self.vocab_size,
            max_seq_length=self.max_seq_length,
            dropout=self.dropout_rate
        )
        
        # Output Processing Layer
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Special token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, 
                src: torch.Tensor,
                tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Sanskrit LLM
        
        Args:
            src: Source sequence [batch_size, src_seq_length]
            tgt: Target sequence [batch_size, tgt_seq_length] (for training)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size = src.size(0)
        
        # Create attention masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_token_id)
            
        # Encoder forward pass
        encoder_output = self.encoder(src, src_mask)
        
        # If target is provided (training mode)
        if tgt is not None:
            if tgt_mask is None:
                tgt_mask = create_combined_mask(tgt, self.pad_token_id)
                
            # Decoder forward pass
            decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
            
            # Output projection to vocabulary
            logits = self.output_projection(decoder_output)
            
            return {
                'logits': logits,
                'encoder_output': encoder_output,
                'decoder_output': decoder_output
            }
        else:
            # Inference mode - return encoder output for further processing
            return {
                'encoder_output': encoder_output
            }
            
    def generate_answer(self, 
                       question: torch.Tensor,
                       max_length: int = 150,
                       temperature: float = 1.0,
                       top_k: int = 50,
                       top_p: float = 0.9) -> torch.Tensor:
        """
        Generate answer for a given question using autoregressive generation
        
        Args:
            question: Input question tokens [batch_size, seq_length]
            max_length: Maximum length of generated answer
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated answer tokens [batch_size, answer_length]
        """
        self.eval()
        with torch.no_grad():
            batch_size = question.size(0)
            device = question.device
            
            # Encode the question
            src_mask = create_padding_mask(question, self.pad_token_id)
            encoder_output = self.encoder(question, src_mask)
            
            # Initialize decoder input with BOS token
            decoder_input = torch.full((batch_size, 1), self.bos_token_id, 
                                     dtype=torch.long, device=device)
            
            # Generate answer token by token
            for _ in range(max_length):
                # Create target mask
                tgt_mask = create_combined_mask(decoder_input, self.pad_token_id)
                
                # Decoder forward pass
                decoder_output = self.decoder(decoder_input, encoder_output, src_mask, tgt_mask)
                
                # Get logits for the last token
                next_token_logits = self.output_projection(decoder_output[:, -1, :])
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k and top-p filtering
                next_token_logits = self._top_k_top_p_filtering(next_token_logits, top_k, top_p)
                
                # Sample next token
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                
                # Append to decoder input
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                # Check if all sequences have generated EOS token
                if torch.all(next_token == self.eos_token_id):
                    break
                    
            return decoder_input[:, 1:]  # Remove BOS token
            
    def _top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Apply top-k and top-p filtering to logits"""
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
            
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
            
        return logits
        
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                    label_smoothing: float = 0.1) -> torch.Tensor:
        """
        Compute cross-entropy loss with label smoothing
        
        Args:
            logits: Model predictions [batch_size, seq_length, vocab_size]
            targets: Target tokens [batch_size, seq_length]
            label_smoothing: Label smoothing factor
            
        Returns:
            Loss tensor
        """
        # Flatten logits and targets
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Ignore padding tokens
        mask = targets != self.pad_token_id
        
        if label_smoothing > 0:
            # Apply label smoothing
            n_class = logits.size(-1)
            one_hot = torch.full_like(logits, label_smoothing / (n_class - 1))
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -one_hot * log_probs
            loss = loss.sum(dim=-1)
        else:
            # Standard cross-entropy loss
            loss = F.cross_entropy(logits, targets, reduction='none')
            
        # Apply mask and compute mean
        loss = loss * mask.float()
        return loss.sum() / mask.sum()
        
    def get_embeddings(self) -> nn.Embedding:
        """Get the embedding layer"""
        return self.encoder.embedding
        
    def set_embeddings(self, embedding_matrix: torch.Tensor):
        """Set pre-trained embeddings"""
        self.encoder.embedding.weight.data = embedding_matrix
        self.decoder.embedding.weight.data = embedding_matrix


class OutputProcessingLayer(nn.Module):
    """
    Output Processing Layer with Linear Transformation and Softmax
    For context-aware answer generation
    """
    
    def __init__(self, d_model: int, vocab_size: int, num_classes: int = None):
        super(OutputProcessingLayer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Linear transformation layers for context-aware answers
        self.context_projection = nn.Linear(d_model, d_model)
        self.answer_projection = nn.Linear(d_model, d_model)
        
        # Final vocabulary projection
        self.vocab_projection = nn.Linear(d_model, vocab_size)
        
        # Classification head for answer type (if needed)
        if num_classes:
            self.classification_head = nn.Linear(d_model, num_classes)
            
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                context_states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of output processing layer
        
        Args:
            hidden_states: Hidden states from decoder [batch_size, seq_length, d_model]
            context_states: Context states from encoder [batch_size, context_length, d_model]
            
        Returns:
            Dictionary containing outputs
        """
        outputs = {}
        
        # Context-aware processing if context is provided
        if context_states is not None:
            # Project context and hidden states
            context_features = self.context_projection(context_states)
            answer_features = self.answer_projection(hidden_states)
            
            # Compute attention between answer and context
            attention_scores = torch.matmul(answer_features, context_features.transpose(-2, -1))
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention to context
            attended_context = torch.matmul(attention_weights, context_features)
            
            # Combine with hidden states
            combined_features = hidden_states + attended_context
            outputs['attention_weights'] = attention_weights
        else:
            combined_features = hidden_states
            
        # Apply dropout
        combined_features = self.dropout(combined_features)
        
        # Generate vocabulary logits
        vocab_logits = self.vocab_projection(combined_features)
        outputs['vocab_logits'] = vocab_logits
        
        # Apply softmax for probability distribution
        vocab_probs = F.softmax(vocab_logits, dim=-1)
        outputs['vocab_probs'] = vocab_probs
        
        # Classification if head exists
        if hasattr(self, 'classification_head'):
            # Pool hidden states for classification
            pooled_features = hidden_states.mean(dim=1)
            class_logits = self.classification_head(pooled_features)
            outputs['class_logits'] = class_logits
            outputs['class_probs'] = F.softmax(class_logits, dim=-1)
            
        return outputs


class QAOutputGenerator(nn.Module):
    """
    Question-Answering Output Generator
    Specialized component for generating contextual Sanskrit answers
    """
    
    def __init__(self, config):
        super(QAOutputGenerator, self).__init__()
        
        self.config = config
        self.d_model = config.HIDDEN_SIZE
        self.vocab_size = config.VOCAB_SIZE
        self.max_answer_length = config.MAX_ANSWER_LENGTH
        self.min_answer_length = config.MIN_ANSWER_LENGTH
        
        # Answer span prediction layers
        self.start_projection = nn.Linear(self.d_model, 1)
        self.end_projection = nn.Linear(self.d_model, 1)
        
        # Answer generation layers
        self.output_processor = OutputProcessingLayer(
            d_model=self.d_model,
            vocab_size=self.vocab_size
        )
        
        # Answer type classifier (extractive vs generative)
        self.answer_type_classifier = nn.Linear(self.d_model, 2)
        
    def forward(self, 
                encoder_output: torch.Tensor,
                decoder_output: torch.Tensor,
                question_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate QA outputs
        
        Args:
            encoder_output: Encoded question/context [batch_size, seq_length, d_model]
            decoder_output: Decoded answer features [batch_size, answer_length, d_model]
            question_mask: Question attention mask
            
        Returns:
            Dictionary containing QA outputs
        """
        outputs = {}
        
        # Answer span prediction (for extractive QA)
        start_logits = self.start_projection(encoder_output).squeeze(-1)
        end_logits = self.end_projection(encoder_output).squeeze(-1)
        
        # Apply mask to span logits
        start_logits = start_logits.masked_fill(~question_mask.squeeze(1).squeeze(1), -1e9)
        end_logits = end_logits.masked_fill(~question_mask.squeeze(1).squeeze(1), -1e9)
        
        outputs['start_logits'] = start_logits
        outputs['end_logits'] = end_logits
        
        # Answer generation (for generative QA)
        generation_outputs = self.output_processor(decoder_output, encoder_output)
        outputs.update(generation_outputs)
        
        # Answer type classification
        pooled_encoder = encoder_output.mean(dim=1)
        answer_type_logits = self.answer_type_classifier(pooled_encoder)
        outputs['answer_type_logits'] = answer_type_logits
        outputs['answer_type_probs'] = F.softmax(answer_type_logits, dim=-1)
        
        return outputs 