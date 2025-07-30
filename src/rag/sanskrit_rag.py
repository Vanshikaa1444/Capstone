"""
Sanskrit Retrieval-Augmented Generation (RAG) System
Combines base Transformer model with retrieval mechanism
to fetch relevant Sanskrit documents before generating responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import faiss
import pickle
from dataclasses import dataclass
from collections import defaultdict
import json
import os

@dataclass
class RetrievedDocument:
    """Data class for retrieved documents"""
    content: str
    score: float
    metadata: Dict[str, Any]
    embedding: np.ndarray
    
@dataclass
class RAGOutput:
    """Data class for RAG outputs"""
    generated_text: str
    retrieved_documents: List[RetrievedDocument]
    retrieval_scores: List[float]
    generation_confidence: float

class SanskritDocumentRetriever:
    """
    Document retriever for Sanskrit texts
    Uses FAISS for efficient similarity search
    """
    
    def __init__(self, config):
        self.config = config
        self.embedding_dim = config.HIDDEN_SIZE
        self.top_k = config.RAG_TOP_K
        self.chunk_size = config.RAG_CHUNK_SIZE
        self.overlap = config.RAG_OVERLAP
        self.threshold = config.RETRIEVAL_THRESHOLD
        
        # FAISS index for document embeddings
        self.index = None
        self.documents = []
        self.document_embeddings = []
        self.document_metadata = []
        
        # Text processing components
        self.document_encoder = None
        
    def initialize_index(self, embedding_dim: int):
        """Initialize FAISS index for document retrieval"""
        # Use inner product similarity (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(embedding_dim)
        print(f"Initialized FAISS index with dimension {embedding_dim}")
        
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: np.ndarray,
                     metadata: List[Dict] = None):
        """
        Add documents and their embeddings to the retriever
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings [num_docs, embedding_dim]
            metadata: List of metadata dictionaries for each document
        """
        if self.index is None:
            self.initialize_index(embeddings.shape[1])
            
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.document_embeddings.extend(embeddings)
        
        if metadata is None:
            metadata = [{"doc_id": i} for i in range(len(documents))]
        self.document_metadata.extend(metadata)
        
        print(f"Added {len(documents)} documents to retriever. Total: {len(self.documents)}")
        
    def chunk_document(self, text: str) -> List[str]:
        """Chunk document into smaller pieces with overlap"""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start += self.chunk_size - self.overlap
            
        return chunks
        
    def retrieve(self, 
                query_embedding: np.ndarray, 
                k: int = None,
                threshold: float = None) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query_embedding: Query embedding [embedding_dim]
            k: Number of documents to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of retrieved documents
        """
        if self.index is None or len(self.documents) == 0:
            return []
            
        k = k or self.top_k
        threshold = threshold or self.threshold
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]  # Remove batch dimension
        indices = indices[0]
        
        # Filter by threshold and create RetrievedDocument objects
        retrieved_docs = []
        for score, idx in zip(scores, indices):
            if score >= threshold and idx < len(self.documents):
                doc = RetrievedDocument(
                    content=self.documents[idx],
                    score=float(score),
                    metadata=self.document_metadata[idx],
                    embedding=self.document_embeddings[idx]
                )
                retrieved_docs.append(doc)
                
        return retrieved_docs
        
    def save_index(self, filepath: str):
        """Save retriever state to file"""
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, f"{filepath}.faiss")
            
        # Save other components
        data = {
            'documents': self.documents,
            'document_embeddings': self.document_embeddings,
            'document_metadata': self.document_metadata,
            'config': {
                'embedding_dim': self.embedding_dim,
                'top_k': self.top_k,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'threshold': self.threshold
            }
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved retriever to {filepath}")
        
    def load_index(self, filepath: str):
        """Load retriever state from file"""
        # Load FAISS index
        if os.path.exists(f"{filepath}.faiss"):
            self.index = faiss.read_index(f"{filepath}.faiss")
            
        # Load other components
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            
        self.documents = data['documents']
        self.document_embeddings = data['document_embeddings']
        self.document_metadata = data['document_metadata']
        
        # Update config
        config = data['config']
        self.embedding_dim = config['embedding_dim']
        self.top_k = config['top_k']
        self.chunk_size = config['chunk_size']
        self.overlap = config['overlap']
        self.threshold = config['threshold']
        
        print(f"Loaded retriever from {filepath}")


class ContextAwareEncoder(nn.Module):
    """
    Context-aware encoder that combines query and retrieved documents
    """
    
    def __init__(self, config):
        super(ContextAwareEncoder, self).__init__()
        
        self.config = config
        self.d_model = config.HIDDEN_SIZE
        self.max_docs = config.RAG_TOP_K
        
        # Document encoding layers
        self.doc_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=self.d_model * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Query-document attention
        self.query_doc_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Document relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        
        # Context fusion layer
        self.context_fusion = nn.Linear(self.d_model * (self.max_docs + 1), self.d_model)
        
    def forward(self, 
                query_embeddings: torch.Tensor,
                doc_embeddings: List[torch.Tensor],
                doc_masks: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Encode query with retrieved document context
        
        Args:
            query_embeddings: [batch_size, query_length, d_model]
            doc_embeddings: List of document embeddings [batch_size, doc_length, d_model]
            doc_masks: List of document attention masks
            
        Returns:
            Dictionary containing encoded outputs
        """
        batch_size = query_embeddings.size(0)
        device = query_embeddings.device
        
        # Pool query for document relevance scoring
        query_pooled = query_embeddings.mean(dim=1)  # [batch_size, d_model]
        
        # Process each document
        processed_docs = []
        relevance_scores = []
        
        for i, doc_emb in enumerate(doc_embeddings):
            # Encode document
            doc_mask = doc_masks[i] if doc_masks else None
            encoded_doc = self.doc_encoder(doc_emb, src_key_padding_mask=doc_mask)
            
            # Pool document
            if doc_mask is not None:
                # Masked pooling
                mask_expanded = (~doc_mask).unsqueeze(-1).float()
                doc_pooled = (encoded_doc * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                doc_pooled = encoded_doc.mean(dim=1)
                
            processed_docs.append(doc_pooled)
            
            # Compute relevance score
            combined = torch.cat([query_pooled, doc_pooled], dim=-1)
            relevance = self.relevance_scorer(combined)
            relevance_scores.append(relevance)
            
        # Stack processed documents
        if processed_docs:
            processed_docs = torch.stack(processed_docs, dim=1)  # [batch_size, num_docs, d_model]
            relevance_scores = torch.cat(relevance_scores, dim=-1)  # [batch_size, num_docs]
        else:
            # No documents retrieved
            processed_docs = torch.zeros(batch_size, 1, self.d_model, device=device)
            relevance_scores = torch.zeros(batch_size, 1, device=device)
            
        # Apply query-document attention
        attended_docs, attention_weights = self.query_doc_attention(
            query_embeddings.mean(dim=1, keepdim=True),  # [batch_size, 1, d_model]
            processed_docs,
            processed_docs
        )
        
        # Combine query and attended documents
        combined_context = torch.cat([
            query_pooled.unsqueeze(1),  # [batch_size, 1, d_model]
            attended_docs  # [batch_size, 1, d_model] or [batch_size, num_docs, d_model]
        ], dim=1)
        
        # Flatten and fuse context
        combined_flat = combined_context.view(batch_size, -1)
        
        # Pad or truncate to expected size
        expected_size = self.d_model * (self.max_docs + 1)
        if combined_flat.size(1) < expected_size:
            padding = torch.zeros(batch_size, expected_size - combined_flat.size(1), device=device)
            combined_flat = torch.cat([combined_flat, padding], dim=1)
        elif combined_flat.size(1) > expected_size:
            combined_flat = combined_flat[:, :expected_size]
            
        context_fused = self.context_fusion(combined_flat)
        
        return {
            'context_embeddings': context_fused,
            'document_relevance': relevance_scores,
            'attention_weights': attention_weights,
            'processed_documents': processed_docs
        }


class SanskritRAGModel(nn.Module):
    """
    Complete Sanskrit RAG Model
    Combines retrieval and generation for enhanced Sanskrit QA
    """
    
    def __init__(self, base_model, config):
        super(SanskritRAGModel, self).__init__()
        
        self.config = config
        self.base_model = base_model
        self.d_model = config.HIDDEN_SIZE
        self.vocab_size = config.VOCAB_SIZE
        
        # Document retriever
        self.retriever = SanskritDocumentRetriever(config)
        
        # Context-aware encoder
        self.context_encoder = ContextAwareEncoder(config)
        
        # RAG-enhanced generation head
        self.rag_generator = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.vocab_size)
        )
        
        # Retrieval confidence estimator
        self.retrieval_confidence = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Generation mode selector (retrieval vs base generation)
        self.mode_selector = nn.Sequential(
            nn.Linear(self.d_model, 2),
            nn.Softmax(dim=-1)
        )
        
    def encode_text_for_retrieval(self, text: str, tokenizer) -> np.ndarray:
        """Encode text for retrieval using base model encoder"""
        # Tokenize text
        tokens = tokenizer.encode(text, max_length=512)
        tokens_tensor = torch.tensor([tokens], device=next(self.parameters()).device)
        
        # Get embeddings from base model
        with torch.no_grad():
            encoder_output = self.base_model.encoder(tokens_tensor)
            # Pool to get document representation
            text_embedding = encoder_output.mean(dim=1).cpu().numpy()
            
        return text_embedding[0]
        
    def retrieve_documents(self, 
                          query: str, 
                          tokenizer,
                          k: int = None) -> List[RetrievedDocument]:
        """Retrieve relevant documents for query"""
        # Encode query
        query_embedding = self.encode_text_for_retrieval(query, tokenizer)
        
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query_embedding, k=k)
        
        return retrieved_docs
        
    def forward(self, 
                query_tokens: torch.Tensor,
                retrieved_docs: Optional[List[RetrievedDocument]] = None,
                target_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of RAG model
        
        Args:
            query_tokens: Input query tokens [batch_size, seq_length]
            retrieved_docs: List of retrieved documents
            target_tokens: Target tokens for training
            
        Returns:
            Dictionary containing model outputs
        """
        # Base model forward pass
        base_outputs = self.base_model(query_tokens, target_tokens)
        base_encoder_output = base_outputs['encoder_output']
        
        if retrieved_docs and len(retrieved_docs) > 0:
            # Process retrieved documents
            doc_embeddings = []
            for doc in retrieved_docs:
                # Convert document to embeddings (simplified - in practice, would tokenize and encode)
                doc_emb = torch.tensor(doc.embedding, device=query_tokens.device).unsqueeze(0)
                doc_emb = doc_emb.expand(query_tokens.size(0), -1, -1)  # Match batch size
                doc_embeddings.append(doc_emb)
                
            # Context-aware encoding
            context_outputs = self.context_encoder(base_encoder_output, doc_embeddings)
            context_embeddings = context_outputs['context_embeddings']
            
            # Combine base and context embeddings
            combined_embeddings = torch.cat([
                base_encoder_output.mean(dim=1),
                context_embeddings
            ], dim=-1)
            
            # RAG generation
            rag_logits = self.rag_generator(combined_embeddings)
            
            # Retrieval confidence
            retrieval_conf = self.retrieval_confidence(context_embeddings)
            
            # Generation mode selection
            mode_probs = self.mode_selector(context_embeddings)
            
            outputs = {
                'rag_logits': rag_logits,
                'base_logits': base_outputs.get('logits'),
                'retrieval_confidence': retrieval_conf,
                'mode_probabilities': mode_probs,
                'document_relevance': context_outputs['document_relevance'],
                'context_embeddings': context_embeddings,
                **base_outputs
            }
        else:
            # No retrieval - use base model only
            outputs = base_outputs
            outputs['retrieval_confidence'] = torch.zeros(query_tokens.size(0), 1, device=query_tokens.device)
            
        return outputs
        
    def generate_with_rag(self, 
                         query: str,
                         tokenizer,
                         max_length: int = 150,
                         temperature: float = 1.0,
                         top_k: int = 50,
                         top_p: float = 0.9,
                         use_retrieval: bool = True) -> RAGOutput:
        """
        Generate response using RAG
        
        Args:
            query: Input query string
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            use_retrieval: Whether to use retrieval
            
        Returns:
            RAGOutput containing generated text and retrieval info
        """
        self.eval()
        with torch.no_grad():
            # Tokenize query
            query_tokens = tokenizer.encode(query, max_length=512)
            query_tensor = torch.tensor([query_tokens], device=next(self.parameters()).device)
            
            retrieved_docs = []
            if use_retrieval:
                # Retrieve relevant documents
                retrieved_docs = self.retrieve_documents(query, tokenizer)
                
            # Forward pass
            outputs = self.forward(query_tensor, retrieved_docs)
            
            # Select logits based on mode (if available)
            if 'rag_logits' in outputs and 'mode_probabilities' in outputs:
                mode_probs = outputs['mode_probabilities'][0]  # [2]
                rag_weight = mode_probs[0].item()
                base_weight = mode_probs[1].item()
                
                # Weighted combination of RAG and base logits
                combined_logits = (rag_weight * outputs['rag_logits'] + 
                                 base_weight * outputs['base_logits'])
            else:
                combined_logits = outputs.get('logits', outputs.get('rag_logits'))
                
            # Generate response token by token (simplified)
            generated_tokens = self._generate_tokens(
                combined_logits, max_length, temperature, top_k, top_p
            )
            
            # Decode generated tokens
            generated_text = tokenizer.decode(generated_tokens[0])
            
            # Get generation confidence
            generation_confidence = outputs.get('retrieval_confidence', torch.tensor([[0.5]]))[0, 0].item()
            
            return RAGOutput(
                generated_text=generated_text,
                retrieved_documents=retrieved_docs,
                retrieval_scores=[doc.score for doc in retrieved_docs],
                generation_confidence=generation_confidence
            )
            
    def _generate_tokens(self, 
                        logits: torch.Tensor,
                        max_length: int,
                        temperature: float,
                        top_k: int,
                        top_p: float) -> torch.Tensor:
        """Simple token generation (would be more sophisticated in practice)"""
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k and top-p filtering
        probs = F.softmax(logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.unsqueeze(0)  # Add batch dimension
        
    def add_documents_to_retriever(self, 
                                  documents: List[str],
                                  tokenizer,
                                  metadata: List[Dict] = None):
        """Add documents to the retriever"""
        # Encode all documents
        embeddings = []
        for doc in documents:
            embedding = self.encode_text_for_retrieval(doc, tokenizer)
            embeddings.append(embedding)
            
        embeddings = np.array(embeddings)
        
        # Add to retriever
        self.retriever.add_documents(documents, embeddings, metadata) 