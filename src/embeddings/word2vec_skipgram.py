"""
Word2Vec Skip-Gram Model for Sanskrit Text
Generates input embeddings for the Transformer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import random
import pickle
from tqdm import tqdm

class SanskritWord2Vec:
    def __init__(self, vector_size: int = 300, window: int = 5, min_count: int = 2, 
                 epochs: int = 100, learning_rate: float = 0.025, negative_samples: int = 5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        
        # Vocabulary and mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = defaultdict(int)
        self.vocab_size = 0
        
        # Model components
        self.center_embeddings = None
        self.context_embeddings = None
        self.model = None
        
        # Training data
        self.training_pairs = []
        
    def build_vocabulary(self, tokenized_texts: List[List[str]]):
        """Build vocabulary from tokenized texts"""
        print("Building vocabulary for Word2Vec...")
        
        # Count word frequencies
        for tokens in tokenized_texts:
            for token in tokens:
                self.word_counts[token] += 1
                
        # Filter by minimum count and build mappings
        vocab_words = [word for word, count in self.word_counts.items() 
                      if count >= self.min_count]
        
        self.vocab_size = len(vocab_words)
        
        for idx, word in enumerate(vocab_words):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            
        print(f"Vocabulary size: {self.vocab_size}")
        
    def generate_training_data(self, tokenized_texts: List[List[str]]):
        """Generate skip-gram training pairs"""
        print("Generating skip-gram training pairs...")
        
        self.training_pairs = []
        
        for tokens in tqdm(tokenized_texts):
            # Filter tokens that are in vocabulary
            valid_tokens = [token for token in tokens if token in self.word_to_idx]
            
            for center_idx, center_word in enumerate(valid_tokens):
                # Define window boundaries
                start = max(0, center_idx - self.window)
                end = min(len(valid_tokens), center_idx + self.window + 1)
                
                # Generate context pairs
                for context_idx in range(start, end):
                    if context_idx != center_idx:
                        context_word = valid_tokens[context_idx]
                        center_word_idx = self.word_to_idx[center_word]
                        context_word_idx = self.word_to_idx[context_word]
                        self.training_pairs.append((center_word_idx, context_word_idx))
                        
        print(f"Generated {len(self.training_pairs)} training pairs")
        
    def get_negative_samples(self, target_word_idx: int, num_samples: int) -> List[int]:
        """Get negative samples for negative sampling"""
        negative_samples = []
        
        # Sample based on word frequency (with smoothing)
        word_probs = np.array([self.word_counts[self.idx_to_word[i]] ** 0.75 
                              for i in range(self.vocab_size)])
        word_probs = word_probs / word_probs.sum()
        
        while len(negative_samples) < num_samples:
            sample = np.random.choice(self.vocab_size, p=word_probs)
            if sample != target_word_idx and sample not in negative_samples:
                negative_samples.append(sample)
                
        return negative_samples


class SkipGramModel(nn.Module):
    """PyTorch Skip-Gram model with negative sampling"""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Center word embeddings (input)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context word embeddings (output) 
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with small random values"""
        init_range = 0.5 / self.embedding_dim
        
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, center_words: torch.Tensor, context_words: torch.Tensor, 
                negative_words: torch.Tensor) -> torch.Tensor:
        """Forward pass with negative sampling"""
        batch_size = center_words.size(0)
        
        # Get embeddings
        center_embeds = self.center_embeddings(center_words)  # [batch_size, embed_dim]
        context_embeds = self.context_embeddings(context_words)  # [batch_size, embed_dim]
        negative_embeds = self.context_embeddings(negative_words)  # [batch_size, neg_samples, embed_dim]
        
        # Positive score
        positive_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        positive_score = torch.sigmoid(positive_score)
        positive_score = torch.clamp(positive_score, min=1e-10, max=1-1e-10)
        positive_loss = -torch.log(positive_score)
        
        # Negative scores
        # Expand center embeddings to match negative samples shape
        center_embeds_expanded = center_embeds.unsqueeze(1)  # [batch_size, 1, embed_dim]
        negative_scores = torch.sum(center_embeds_expanded * negative_embeds, dim=2)  # [batch_size, neg_samples]
        negative_scores = torch.sigmoid(negative_scores)
        negative_scores = torch.clamp(negative_scores, min=1e-10, max=1-1e-10)
        negative_loss = -torch.sum(torch.log(1 - negative_scores), dim=1)  # [batch_size]
        
        # Total loss
        loss = positive_loss + negative_loss
        return loss.mean()
        
    def get_embeddings(self) -> torch.Tensor:
        """Get the trained center word embeddings"""
        return self.center_embeddings.weight.data


class SanskritWord2VecTrainer:
    """Trainer class for Sanskrit Word2Vec model"""
    
    def __init__(self, word2vec_model: SanskritWord2Vec, device: str = 'cpu'):
        self.word2vec_model = word2vec_model
        self.device = device
        self.model = None
        self.optimizer = None
        
    def train(self, tokenized_texts: List[List[str]], batch_size: int = 1024):
        """Train the Word2Vec model"""
        # Build vocabulary and generate training data
        self.word2vec_model.build_vocabulary(tokenized_texts)
        self.word2vec_model.generate_training_data(tokenized_texts)
        
        # Initialize PyTorch model
        self.model = SkipGramModel(
            vocab_size=self.word2vec_model.vocab_size,
            embedding_dim=self.word2vec_model.vector_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.word2vec_model.learning_rate)
        
        print(f"Training Word2Vec model for {self.word2vec_model.epochs} epochs...")
        
        # Training loop
        for epoch in range(self.word2vec_model.epochs):
            total_loss = 0
            batch_count = 0
            
            # Shuffle training pairs
            random.shuffle(self.word2vec_model.training_pairs)
            
            # Process in batches
            for i in tqdm(range(0, len(self.word2vec_model.training_pairs), batch_size), 
                         desc=f"Epoch {epoch+1}/{self.word2vec_model.epochs}"):
                
                batch_pairs = self.word2vec_model.training_pairs[i:i+batch_size]
                
                if len(batch_pairs) == 0:
                    continue
                    
                # Prepare batch data
                center_words = []
                context_words = []
                negative_words_batch = []
                
                for center_idx, context_idx in batch_pairs:
                    center_words.append(center_idx)
                    context_words.append(context_idx)
                    
                    # Get negative samples
                    negative_samples = self.word2vec_model.get_negative_samples(
                        context_idx, self.word2vec_model.negative_samples
                    )
                    negative_words_batch.append(negative_samples)
                
                # Convert to tensors
                center_words = torch.tensor(center_words, dtype=torch.long).to(self.device)
                context_words = torch.tensor(context_words, dtype=torch.long).to(self.device)
                negative_words = torch.tensor(negative_words_batch, dtype=torch.long).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                loss = self.model(center_words, context_words, negative_words)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
        print("Word2Vec training completed!")
        
        # Store embeddings in the word2vec model
        self.word2vec_model.center_embeddings = self.model.get_embeddings()
        
    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a specific word"""
        if word not in self.word2vec_model.word_to_idx:
            return None
            
        word_idx = self.word2vec_model.word_to_idx[word]
        if self.word2vec_model.center_embeddings is not None:
            return self.word2vec_model.center_embeddings[word_idx].cpu().numpy()
        return None
        
    def get_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words using cosine similarity"""
        if word not in self.word2vec_model.word_to_idx:
            return []
            
        word_idx = self.word2vec_model.word_to_idx[word]
        word_embedding = self.word2vec_model.center_embeddings[word_idx]
        
        # Calculate cosine similarity with all words
        embeddings = self.word2vec_model.center_embeddings
        similarities = torch.cosine_similarity(word_embedding.unsqueeze(0), embeddings)
        
        # Get top-k most similar words
        top_indices = torch.topk(similarities, top_k + 1).indices[1:]  # Exclude the word itself
        
        similar_words = []
        for idx in top_indices:
            similar_word = self.word2vec_model.idx_to_word[idx.item()]
            similarity_score = similarities[idx].item()
            similar_words.append((similar_word, similarity_score))
            
        return similar_words
        
    def save_model(self, filepath: str):
        """Save the trained Word2Vec model"""
        model_data = {
            'word_to_idx': self.word2vec_model.word_to_idx,
            'idx_to_word': self.word2vec_model.idx_to_word,
            'word_counts': dict(self.word2vec_model.word_counts),
            'vocab_size': self.word2vec_model.vocab_size,
            'vector_size': self.word2vec_model.vector_size,
            'window': self.word2vec_model.window,
            'min_count': self.word2vec_model.min_count,
            'embeddings': self.word2vec_model.center_embeddings.cpu().numpy() if self.word2vec_model.center_embeddings is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Word2Vec model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load a pre-trained Word2Vec model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.word2vec_model.word_to_idx = model_data['word_to_idx']
        self.word2vec_model.idx_to_word = model_data['idx_to_word']
        self.word2vec_model.word_counts = defaultdict(int, model_data['word_counts'])
        self.word2vec_model.vocab_size = model_data['vocab_size']
        self.word2vec_model.vector_size = model_data['vector_size']
        self.word2vec_model.window = model_data['window']
        self.word2vec_model.min_count = model_data['min_count']
        
        if model_data['embeddings'] is not None:
            self.word2vec_model.center_embeddings = torch.tensor(model_data['embeddings'])
            
        print(f"Word2Vec model loaded from {filepath}")


def create_embedding_matrix(word2vec_model: SanskritWord2Vec, tokenizer_vocab: Dict[str, int], 
                           embedding_dim: int) -> torch.Tensor:
    """Create embedding matrix for transformer model from Word2Vec embeddings"""
    vocab_size = len(tokenizer_vocab)
    embedding_matrix = torch.randn(vocab_size, embedding_dim) * 0.01
    
    found_words = 0
    for word, idx in tokenizer_vocab.items():
        if word in word2vec_model.word_to_idx:
            w2v_idx = word2vec_model.word_to_idx[word]
            if word2vec_model.center_embeddings is not None:
                embedding_matrix[idx] = word2vec_model.center_embeddings[w2v_idx]
                found_words += 1
                
    print(f"Found Word2Vec embeddings for {found_words}/{vocab_size} tokens")
    return embedding_matrix 