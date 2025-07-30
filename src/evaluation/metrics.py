"""
Evaluation Metrics for Sanskrit LLM
Includes BLEU score, retrieval accuracy, latency, and perplexity calculations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time
from collections import Counter, defaultdict
import math
from sacrebleu import BLEU
import json
from dataclasses import dataclass

@dataclass
class EvaluationResults:
    """Data class for evaluation results"""
    bleu_score: float
    bleu_scores_breakdown: Dict[str, float]
    retrieval_accuracy: float
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    latency: float
    perplexity: float
    exact_match: float
    semantic_similarity: float
    navarasa_accuracy: float
    generation_quality: Dict[str, float]

class BLEUEvaluator:
    """BLEU score evaluator for Sanskrit text generation"""
    
    def __init__(self, weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)):
        self.weights = weights
        self.bleu_scorer = BLEU()
        
    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Compute BLEU scores for predictions against references
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be multiple references per prediction)
            
        Returns:
            Dictionary containing BLEU scores
        """
        if not predictions or not references:
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
            
        # Compute overall BLEU score using sacrebleu
        bleu_score = self.bleu_scorer.corpus_score(predictions, references).score
        
        # Compute individual n-gram BLEU scores
        bleu_scores = {}
        for n in range(1, 5):
            n_gram_weights = tuple([1.0/n if i < n else 0.0 for i in range(4)])
            n_bleu_scorer = BLEU(effective_order=True)
            n_score = n_bleu_scorer.corpus_score(predictions, references).score
            bleu_scores[f"bleu_{n}"] = n_score
            
        bleu_scores["bleu"] = bleu_score
        
        return bleu_scores
        
    def compute_sentence_bleu(self, prediction: str, reference: str) -> float:
        """Compute BLEU score for a single sentence pair"""
        return self.bleu_scorer.sentence_score(prediction, [reference]).score


class RetrievalEvaluator:
    """Evaluator for retrieval accuracy in RAG system"""
    
    def __init__(self):
        pass
        
    def compute_retrieval_metrics(self, 
                                retrieved_docs: List[List[str]],
                                relevant_docs: List[List[str]],
                                k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Compute retrieval metrics
        
        Args:
            retrieved_docs: List of retrieved document IDs for each query
            relevant_docs: List of relevant document IDs for each query
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary containing retrieval metrics
        """
        metrics = {}
        
        for k in k_values:
            precision_at_k = self._precision_at_k(retrieved_docs, relevant_docs, k)
            recall_at_k = self._recall_at_k(retrieved_docs, relevant_docs, k)
            f1_at_k = self._f1_score(precision_at_k, recall_at_k)
            
            metrics[f"precision@{k}"] = precision_at_k
            metrics[f"recall@{k}"] = recall_at_k
            metrics[f"f1@{k}"] = f1_at_k
            
        # Overall metrics
        metrics["mean_reciprocal_rank"] = self._mean_reciprocal_rank(retrieved_docs, relevant_docs)
        metrics["ndcg@5"] = self._ndcg_at_k(retrieved_docs, relevant_docs, 5)
        metrics["ndcg@10"] = self._ndcg_at_k(retrieved_docs, relevant_docs, 10)
        
        return metrics
        
    def _precision_at_k(self, retrieved: List[List[str]], relevant: List[List[str]], k: int) -> float:
        """Compute Precision@K"""
        precisions = []
        
        for retr, rel in zip(retrieved, relevant):
            retr_k = retr[:k]
            relevant_set = set(rel)
            
            if len(retr_k) == 0:
                precisions.append(0.0)
            else:
                precision = len(set(retr_k) & relevant_set) / len(retr_k)
                precisions.append(precision)
                
        return np.mean(precisions)
        
    def _recall_at_k(self, retrieved: List[List[str]], relevant: List[List[str]], k: int) -> float:
        """Compute Recall@K"""
        recalls = []
        
        for retr, rel in zip(retrieved, relevant):
            retr_k = retr[:k]
            relevant_set = set(rel)
            
            if len(relevant_set) == 0:
                recalls.append(0.0)
            else:
                recall = len(set(retr_k) & relevant_set) / len(relevant_set)
                recalls.append(recall)
                
        return np.mean(recalls)
        
    def _f1_score(self, precision: float, recall: float) -> float:
        """Compute F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
        
    def _mean_reciprocal_rank(self, retrieved: List[List[str]], relevant: List[List[str]]) -> float:
        """Compute Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for retr, rel in zip(retrieved, relevant):
            relevant_set = set(rel)
            reciprocal_rank = 0.0
            
            for i, doc in enumerate(retr):
                if doc in relevant_set:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
                    
            reciprocal_ranks.append(reciprocal_rank)
            
        return np.mean(reciprocal_ranks)
        
    def _ndcg_at_k(self, retrieved: List[List[str]], relevant: List[List[str]], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at K"""
        ndcg_scores = []
        
        for retr, rel in zip(retrieved, relevant):
            retr_k = retr[:k]
            relevant_set = set(rel)
            
            # Compute DCG
            dcg = 0.0
            for i, doc in enumerate(retr_k):
                if doc in relevant_set:
                    dcg += 1.0 / math.log2(i + 2)
                    
            # Compute IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(len(relevant_set), k)):
                idcg += 1.0 / math.log2(i + 2)
                
            # Compute NDCG
            if idcg == 0:
                ndcg_scores.append(0.0)
            else:
                ndcg_scores.append(dcg / idcg)
                
        return np.mean(ndcg_scores)


class PerplexityEvaluator:
    """Perplexity evaluator for language modeling"""
    
    def __init__(self):
        pass
        
    def compute_perplexity(self, model: nn.Module, data_loader, device: str = 'cuda') -> float:
        """
        Compute perplexity on a dataset
        
        Args:
            model: Language model
            data_loader: DataLoader for evaluation data
            device: Device to run evaluation on
            
        Returns:
            Perplexity score
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in data_loader:
                src = batch['input_ids'].to(device)
                tgt = batch['target_ids'].to(device)
                
                # Forward pass
                outputs = model(src, tgt)
                logits = outputs['logits']
                
                # Compute loss
                loss = model.compute_loss(logits, tgt)
                
                # Accumulate loss and token count
                total_loss += loss.item() * src.size(0)
                total_tokens += src.size(0) * src.size(1)
                
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity


class LatencyEvaluator:
    """Latency evaluator for response generation"""
    
    def __init__(self):
        self.latencies = []
        
    def measure_generation_latency(self, model, query: str, tokenizer, **generation_kwargs) -> Tuple[str, float]:
        """
        Measure generation latency
        
        Args:
            model: Language model
            query: Input query
            tokenizer: Tokenizer
            **generation_kwargs: Generation parameters
            
        Returns:
            Tuple of (generated_text, latency_in_seconds)
        """
        start_time = time.time()
        
        # Generate response
        generated_text = model.generate_answer(query, tokenizer, **generation_kwargs)
        
        end_time = time.time()
        latency = end_time - start_time
        
        self.latencies.append(latency)
        
        return generated_text, latency
        
    def get_latency_statistics(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latencies:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            
        return {
            "mean": np.mean(self.latencies),
            "median": np.median(self.latencies),
            "std": np.std(self.latencies),
            "min": np.min(self.latencies),
            "max": np.max(self.latencies)
        }


class SemanticSimilarityEvaluator:
    """Semantic similarity evaluator"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def compute_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute semantic similarity between predictions and references
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Average semantic similarity score
        """
        similarities = []
        
        for pred, ref in zip(predictions, references):
            similarity = self._cosine_similarity(pred, ref)
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        # Encode texts
        tokens1 = self.tokenizer.encode(text1, max_length=512)
        tokens2 = self.tokenizer.encode(text2, max_length=512)
        
        tokens1_tensor = torch.tensor([tokens1], device=next(self.model.parameters()).device)
        tokens2_tensor = torch.tensor([tokens2], device=next(self.model.parameters()).device)
        
        with torch.no_grad():
            # Get embeddings
            emb1 = self.model.encoder(tokens1_tensor).mean(dim=1)
            emb2 = self.model.encoder(tokens2_tensor).mean(dim=1)
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(emb1, emb2).item()
            
        return similarity


class NavaraSentimentEvaluator:
    """Evaluator for Navarasa sentiment accuracy"""
    
    def __init__(self, emotion_names: List[str]):
        self.emotion_names = emotion_names
        
    def compute_sentiment_accuracy(self, 
                                 predicted_emotions: List[int],
                                 true_emotions: List[int]) -> Dict[str, float]:
        """
        Compute sentiment classification accuracy
        
        Args:
            predicted_emotions: List of predicted emotion indices
            true_emotions: List of true emotion indices
            
        Returns:
            Dictionary containing accuracy metrics
        """
        if len(predicted_emotions) != len(true_emotions):
            raise ValueError("Predicted and true emotions must have same length")
            
        # Overall accuracy
        correct = sum(p == t for p, t in zip(predicted_emotions, true_emotions))
        accuracy = correct / len(predicted_emotions)
        
        # Per-emotion metrics
        emotion_metrics = {}
        for i, emotion in enumerate(self.emotion_names):
            true_positives = sum(p == i and t == i for p, t in zip(predicted_emotions, true_emotions))
            predicted_positives = sum(p == i for p in predicted_emotions)
            actual_positives = sum(t == i for t in true_emotions)
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
            recall = true_positives / actual_positives if actual_positives > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            emotion_metrics[f"{emotion}_precision"] = precision
            emotion_metrics[f"{emotion}_recall"] = recall
            emotion_metrics[f"{emotion}_f1"] = f1
            
        return {
            "overall_accuracy": accuracy,
            **emotion_metrics
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize individual evaluators
        self.bleu_evaluator = BLEUEvaluator(config.BLEU_WEIGHTS)
        self.retrieval_evaluator = RetrievalEvaluator()
        self.perplexity_evaluator = PerplexityEvaluator()
        self.latency_evaluator = LatencyEvaluator()
        self.semantic_evaluator = SemanticSimilarityEvaluator(model, tokenizer)
        self.sentiment_evaluator = NavaraSentimentEvaluator(config.NAVARASA_EMOTIONS)
        
    def evaluate_model(self, 
                      test_dataset,
                      retrieval_data: Optional[Dict] = None,
                      sentiment_data: Optional[Dict] = None) -> EvaluationResults:
        """
        Comprehensive model evaluation
        
        Args:
            test_dataset: Test dataset
            retrieval_data: Data for retrieval evaluation
            sentiment_data: Data for sentiment evaluation
            
        Returns:
            EvaluationResults object
        """
        predictions = []
        references = []
        generation_latencies = []
        
        # Generate predictions and measure latency
        for sample in test_dataset:
            query = sample['query']
            reference = sample['reference']
            
            # Generate prediction with latency measurement
            start_time = time.time()
            if hasattr(self.model, 'generate_with_rag'):
                # RAG model
                rag_output = self.model.generate_with_rag(
                    query, self.tokenizer,
                    max_length=self.config.MAX_ANSWER_LENGTH,
                    temperature=1.0
                )
                prediction = rag_output.generated_text
            else:
                # Base model
                query_tokens = torch.tensor([self.tokenizer.encode(query)], 
                                          device=next(self.model.parameters()).device)
                prediction_tokens = self.model.generate_answer(query_tokens)
                prediction = self.tokenizer.decode(prediction_tokens[0])
                
            end_time = time.time()
            latency = end_time - start_time
            
            predictions.append(prediction)
            references.append([reference])  # BLEU expects list of references
            generation_latencies.append(latency)
            
        # Compute BLEU scores
        bleu_scores = self.bleu_evaluator.compute_bleu(predictions, references)
        
        # Compute semantic similarity
        semantic_similarity = self.semantic_evaluator.compute_semantic_similarity(
            predictions, [ref[0] for ref in references]
        )
        
        # Compute exact match
        exact_matches = sum(pred.strip() == ref[0].strip() 
                          for pred, ref in zip(predictions, references))
        exact_match_score = exact_matches / len(predictions)
        
        # Compute perplexity
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
        perplexity = self.perplexity_evaluator.compute_perplexity(
            self.model, test_loader, device=next(self.model.parameters()).device
        )
        
        # Compute latency statistics
        avg_latency = np.mean(generation_latencies)
        
        # Compute retrieval metrics (if available)
        retrieval_metrics = {"precision@5": 0.0, "recall@5": 0.0, "f1@5": 0.0}
        if retrieval_data:
            retrieval_metrics = self.retrieval_evaluator.compute_retrieval_metrics(
                retrieval_data['retrieved'], retrieval_data['relevant']
            )
            
        # Compute sentiment metrics (if available)
        sentiment_accuracy = 0.0
        if sentiment_data:
            sentiment_metrics = self.sentiment_evaluator.compute_sentiment_accuracy(
                sentiment_data['predicted'], sentiment_data['true']
            )
            sentiment_accuracy = sentiment_metrics['overall_accuracy']
            
        # Generation quality metrics
        generation_quality = self._compute_generation_quality(predictions, references)
        
        return EvaluationResults(
            bleu_score=bleu_scores['bleu'],
            bleu_scores_breakdown=bleu_scores,
            retrieval_accuracy=retrieval_metrics.get('precision@5', 0.0),
            retrieval_precision=retrieval_metrics.get('precision@5', 0.0),
            retrieval_recall=retrieval_metrics.get('recall@5', 0.0),
            retrieval_f1=retrieval_metrics.get('f1@5', 0.0),
            latency=avg_latency,
            perplexity=perplexity,
            exact_match=exact_match_score,
            semantic_similarity=semantic_similarity,
            navarasa_accuracy=sentiment_accuracy,
            generation_quality=generation_quality
        )
        
    def _compute_generation_quality(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute additional generation quality metrics"""
        # Average length
        avg_pred_length = np.mean([len(pred.split()) for pred in predictions])
        avg_ref_length = np.mean([len(ref[0].split()) for ref in references])
        
        # Length ratio
        length_ratio = avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0.0
        
        # Vocabulary diversity (unique words / total words)
        all_words = ' '.join(predictions).split()
        vocab_diversity = len(set(all_words)) / len(all_words) if all_words else 0.0
        
        return {
            "avg_prediction_length": avg_pred_length,
            "avg_reference_length": avg_ref_length,
            "length_ratio": length_ratio,
            "vocabulary_diversity": vocab_diversity
        }
        
    def save_evaluation_results(self, results: EvaluationResults, filepath: str):
        """Save evaluation results to file"""
        results_dict = {
            "bleu_score": results.bleu_score,
            "bleu_scores_breakdown": results.bleu_scores_breakdown,
            "retrieval_accuracy": results.retrieval_accuracy,
            "retrieval_precision": results.retrieval_precision,
            "retrieval_recall": results.retrieval_recall,
            "retrieval_f1": results.retrieval_f1,
            "latency": results.latency,
            "perplexity": results.perplexity,
            "exact_match": results.exact_match,
            "semantic_similarity": results.semantic_similarity,
            "navarasa_accuracy": results.navarasa_accuracy,
            "generation_quality": results.generation_quality
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        print(f"Evaluation results saved to {filepath}") 