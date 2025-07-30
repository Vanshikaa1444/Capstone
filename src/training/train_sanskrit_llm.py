"""
Main Training Script for Sanskrit LLM
Combines all components to train the complete Sanskrit Language Model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all components
from config.model_config import SanskritLLMConfig
from src.tokenizer.sanskrit_tokenizer import SanskritTokenizer
from src.embeddings.word2vec_skipgram import SanskritWord2Vec, SanskritWord2VecTrainer, create_embedding_matrix
from src.models.sanskrit_llm import SanskritLLM, QAOutputGenerator
from src.sentiment.navarasa_sentiment import NavaraSentimentLayer, EmotionAwareResponseGenerator
from src.rag.sanskrit_rag import SanskritRAGModel
from src.data.data_processor import SanskritDatasetBuilder
from src.evaluation.metrics import ComprehensiveEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SanskritDataset(Dataset):
    """PyTorch Dataset for Sanskrit LLM training"""
    
    def __init__(self, questions: List[str], answers: List[str], contexts: List[str], 
                 tokenizer: SanskritTokenizer, max_length: int = 512):
        self.questions = questions
        self.answers = answers
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        context = self.contexts[idx]
        
        # Tokenize inputs
        question_tokens = self.tokenizer.encode(question, max_length=self.max_length)
        answer_tokens = self.tokenizer.encode(answer, max_length=self.max_length)
        context_tokens = self.tokenizer.encode(context, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(question_tokens, dtype=torch.long),
            'target_ids': torch.tensor(answer_tokens, dtype=torch.long),
            'context_ids': torch.tensor(context_tokens, dtype=torch.long),
            'question_text': question,
            'answer_text': answer,
            'context_text': context
        }

class SanskritLLMTrainer:
    """Main trainer class for Sanskrit LLM"""
    
    def __init__(self, config: SanskritLLMConfig):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.word2vec_model = None
        self.model = None
        self.rag_model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Setup directories
        self.setup_directories()
        
    def setup_directories(self):
        """Setup necessary directories"""
        directories = [
            self.config.MODEL_SAVE_DIR,
            self.config.LOG_DIR,
            self.config.PROCESSED_DATA_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def build_tokenizer(self, texts: List[str]):
        """Build and train the Sanskrit tokenizer"""
        logger.info("Building Sanskrit tokenizer...")
        
        self.tokenizer = SanskritTokenizer(
            vocab_size=self.config.VOCAB_SIZE,
            special_tokens=self.config.SPECIAL_TOKENS
        )
        
        # Build vocabulary
        self.tokenizer.build_vocabulary(texts, min_freq=self.config.WORD2VEC_MIN_COUNT)
        
        # Save tokenizer
        tokenizer_path = os.path.join(self.config.MODEL_SAVE_DIR, 'tokenizer.pkl')
        self.tokenizer.save(tokenizer_path)
        
        logger.info(f"Tokenizer built with vocabulary size: {self.tokenizer.get_vocab_size()}")
        
    def train_word2vec(self, texts: List[str]):
        """Train Word2Vec embeddings"""
        logger.info("Training Word2Vec embeddings...")
        
        # Tokenize texts for Word2Vec
        tokenized_texts = []
        for text in texts:
            processed_tokens = self.tokenizer.process_dcs_text(text)
            tokens = [token for token, _ in processed_tokens]
            tokenized_texts.append(tokens)
            
        # Initialize Word2Vec model
        self.word2vec_model = SanskritWord2Vec(
            vector_size=self.config.WORD2VEC_VECTOR_SIZE,
            window=self.config.WORD2VEC_WINDOW,
            min_count=self.config.WORD2VEC_MIN_COUNT,
            epochs=self.config.WORD2VEC_EPOCHS
        )
        
        # Train Word2Vec
        trainer = SanskritWord2VecTrainer(self.word2vec_model, device=str(self.device))
        trainer.train(tokenized_texts)
        
        # Save Word2Vec model
        w2v_path = os.path.join(self.config.MODEL_SAVE_DIR, 'word2vec.pkl')
        trainer.save_model(w2v_path)
        
        logger.info("Word2Vec training completed")
        
    def build_model(self):
        """Build the main Sanskrit LLM model"""
        logger.info("Building Sanskrit LLM model...")
        
        # Initialize main model
        self.model = SanskritLLM(self.config).to(self.device)
        
        # Set pre-trained embeddings if Word2Vec is available
        if self.word2vec_model is not None:
            embedding_matrix = create_embedding_matrix(
                self.word2vec_model,
                self.tokenizer.word_to_id,
                self.config.HIDDEN_SIZE
            )
            self.model.set_embeddings(embedding_matrix)
            
        # Initialize RAG model
        self.rag_model = SanskritRAGModel(self.model, self.config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model built with {total_params:,} parameters ({trainable_params:,} trainable)")
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Parameters to optimize
        params = list(self.model.parameters()) + list(self.rag_model.parameters())
        
        # Optimizer
        if self.config.OPTIMIZER.lower() == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.config.LEARNING_RATE,
                betas=(self.config.BETA1, self.config.BETA2),
                eps=self.config.EPSILON,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")
            
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.MAX_EPOCHS,
            eta_min=self.config.LEARNING_RATE * 0.01
        )
        
    def prepare_data(self):
        """Prepare training data"""
        logger.info("Preparing training data...")
        
        # Build dataset
        dataset_builder = SanskritDatasetBuilder(self.config)
        dataset = dataset_builder.build_complete_dataset()
        
        # Build tokenizer
        self.build_tokenizer(dataset.texts)
        
        # Train Word2Vec
        self.train_word2vec(dataset.texts)
        
        # Create train/val/test splits
        splits = dataset_builder.create_train_val_test_split(
            dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # Create datasets
        train_dataset = SanskritDataset(
            splits['train'].questions,
            splits['train'].answers,
            splits['train'].contexts,
            self.tokenizer,
            self.config.MAX_SEQ_LENGTH
        )
        
        val_dataset = SanskritDataset(
            splits['val'].questions,
            splits['val'].answers,
            splits['val'].contexts,
            self.tokenizer,
            self.config.MAX_SEQ_LENGTH
        )
        
        test_dataset = SanskritDataset(
            splits['test'].questions,
            splits['test'].answers,
            splits['test'].contexts,
            self.tokenizer,
            self.config.MAX_SEQ_LENGTH
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.rag_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.MAX_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, target_ids)
            
            # Compute loss
            loss = self.model.compute_loss(
                outputs['logits'], 
                target_ids,
                label_smoothing=self.config.LABEL_SMOOTHING
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.rag_model.parameters()),
                self.config.GRADIENT_CLIP_NORM
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to wandb
            if self.global_step % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
                
            # Save checkpoint
            if self.global_step % self.config.SAVE_EVERY_N_STEPS == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}')
                
            # Evaluation
            if self.global_step % self.config.EVAL_EVERY_N_STEPS == 0:
                val_metrics = self.evaluate()
                logger.info(f"Step {self.global_step} validation metrics: {val_metrics}")
                
                # Log validation metrics to wandb
                wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
                
                # Save best model
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.save_checkpoint('best_model')
                    
                self.model.train()
                self.rag_model.train()
                
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        self.rag_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, target_ids)
                
                # Compute loss
                loss = self.model.compute_loss(outputs['logits'], target_ids)
                
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
        
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'rag_model_state_dict': self.rag_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.config.MODEL_SAVE_DIR, f'{name}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.rag_model.load_state_dict(checkpoint['rag_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
    def train(self):
        """Main training loop"""
        logger.info("Starting Sanskrit LLM training...")
        
        # Prepare data and model
        self.prepare_data()
        self.build_model()
        self.setup_optimizer()
        
        # Initialize wandb
        wandb.init(
            project="sanskrit-llm",
            config=self.config.__dict__,
            name=f"sanskrit_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Training loop
        for epoch in range(self.config.MAX_EPOCHS):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.evaluate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log epoch metrics
            epoch_metrics = {
                'train/epoch_loss': train_metrics['loss'],
                'val/epoch_loss': val_metrics['loss'],
                'val/epoch_perplexity': val_metrics['perplexity'],
                'epoch': epoch
            }
            
            wandb.log(epoch_metrics)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.MAX_EPOCHS} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Perplexity: {val_metrics['perplexity']:.4f}"
            )
            
            # Save checkpoint
            self.save_checkpoint(f'epoch_{epoch+1}')
            
        logger.info("Training completed!")
        
        # Final evaluation
        self.final_evaluation()
        
    def final_evaluation(self):
        """Comprehensive final evaluation"""
        logger.info("Running final evaluation...")
        
        # Load best model
        best_model_path = os.path.join(self.config.MODEL_SAVE_DIR, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
            
        # Create evaluator
        evaluator = ComprehensiveEvaluator(self.rag_model, self.tokenizer, self.config)
        
        # Convert test loader to required format
        test_samples = []
        for batch in self.test_loader:
            for i in range(len(batch['question_text'])):
                test_samples.append({
                    'query': batch['question_text'][i],
                    'reference': batch['answer_text'][i]
                })
                
        # Run evaluation
        results = evaluator.evaluate_model(test_samples)
        
        # Log results
        final_metrics = {
            'final/bleu_score': results.bleu_score,
            'final/perplexity': results.perplexity,
            'final/exact_match': results.exact_match,
            'final/semantic_similarity': results.semantic_similarity,
            'final/latency': results.latency
        }
        
        wandb.log(final_metrics)
        
        # Save results
        results_path = os.path.join(self.config.LOG_DIR, 'final_evaluation_results.json')
        evaluator.save_evaluation_results(results, results_path)
        
        logger.info(f"Final evaluation completed. Results saved to {results_path}")
        logger.info(f"Final BLEU Score: {results.bleu_score:.4f}")
        logger.info(f"Final Perplexity: {results.perplexity:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Sanskrit LLM')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    
    args = parser.parse_args()
    
    # Load config
    config = SanskritLLMConfig()
    
    # Initialize trainer
    trainer = SanskritLLMTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Run training or evaluation
    if args.eval_only:
        trainer.final_evaluation()
    else:
        trainer.train()

if __name__ == "__main__":
    main() 