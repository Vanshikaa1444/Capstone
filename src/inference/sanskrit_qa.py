"""
Sanskrit Question-Answering System
Interactive interface for the trained Sanskrit LLM
"""

import os
import sys
import torch
import argparse
from typing import Dict, List, Optional
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.model_config import SanskritLLMConfig
from src.tokenizer.sanskrit_tokenizer import SanskritTokenizer
from src.models.sanskrit_llm import SanskritLLM
from src.rag.sanskrit_rag import SanskritRAGModel, RAGOutput
from src.sentiment.navarasa_sentiment import NavaraSentimentLayer

class SanskritQASystem:
    """
    Complete Sanskrit Question-Answering System
    Provides an interface to interact with the trained Sanskrit LLM
    """
    
    def __init__(self, model_path: str, config: SanskritLLMConfig = None):
        self.model_path = model_path
        self.config = config or SanskritLLMConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.rag_model = None
        self.sentiment_layer = None
        
        # Load the trained model
        self.load_model()
        
    def load_model(self):
        """Load the trained model and components"""
        print(f"Loading Sanskrit LLM from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load tokenizer
        tokenizer_path = os.path.join(os.path.dirname(self.model_path), 'tokenizer.pkl')
        if os.path.exists(tokenizer_path):
            self.tokenizer = SanskritTokenizer()
            self.tokenizer.load(tokenizer_path)
        else:
            print("Warning: Tokenizer not found, creating new one")
            self.tokenizer = SanskritTokenizer(
                vocab_size=self.config.VOCAB_SIZE,
                special_tokens=self.config.SPECIAL_TOKENS
            )
            
        # Initialize and load models
        self.model = SanskritLLM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.rag_model = SanskritRAGModel(self.model, self.config).to(self.device)
        self.rag_model.load_state_dict(checkpoint['rag_model_state_dict'])
        
        # Initialize sentiment layer
        self.sentiment_layer = NavaraSentimentLayer(self.config).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        self.rag_model.eval()
        self.sentiment_layer.eval()
        
        print("Model loaded successfully!")
        
    def answer_question(self, 
                       question: str,
                       use_rag: bool = True,
                       max_length: int = 150,
                       temperature: float = 0.8,
                       top_k: int = 50,
                       top_p: float = 0.9) -> Dict:
        """
        Answer a Sanskrit-related question
        
        Args:
            question: The question to answer
            use_rag: Whether to use RAG for enhanced answers
            max_length: Maximum length of generated answer
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing answer and metadata
        """
        if not question.strip():
            return {"error": "Please provide a non-empty question"}
            
        try:
            with torch.no_grad():
                if use_rag and hasattr(self.rag_model, 'generate_with_rag'):
                    # Use RAG model
                    rag_output = self.rag_model.generate_with_rag(
                        question,
                        self.tokenizer,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        use_retrieval=True
                    )
                    
                    return {
                        "question": question,
                        "answer": rag_output.generated_text,
                        "retrieved_documents": [
                            {
                                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                                "score": doc.score,
                                "metadata": doc.metadata
                            }
                            for doc in rag_output.retrieved_documents
                        ],
                        "generation_confidence": rag_output.generation_confidence,
                        "method": "RAG"
                    }
                else:
                    # Use base model
                    question_tokens = torch.tensor(
                        [self.tokenizer.encode(question, max_length=512)],
                        device=self.device
                    )
                    
                    answer_tokens = self.model.generate_answer(
                        question_tokens,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    answer = self.tokenizer.decode(answer_tokens[0])
                    
                    return {
                        "question": question,
                        "answer": answer,
                        "method": "Base Model"
                    }
                    
        except Exception as e:
            return {"error": f"Error generating answer: {str(e)}"}
            
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze the Navarasa sentiment of the given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not text.strip():
            return {"error": "Please provide non-empty text"}
            
        try:
            # Tokenize text
            tokens = self.tokenizer.encode(text, max_length=512)
            tokens_tensor = torch.tensor([tokens], device=self.device)
            
            with torch.no_grad():
                # Get encoder output
                encoder_output = self.model.encoder(tokens_tensor)
                
                # Analyze sentiment
                sentiment_output = self.sentiment_layer(encoder_output)
                
                # Get emotion probabilities
                emotion_probs = sentiment_output['emotion_probs'][0].cpu().numpy()
                dominant_emotion_idx = sentiment_output['dominant_emotion'][0].item()
                emotion_intensity = sentiment_output['emotion_intensity'][0].item()
                
                # Create response
                emotions_dict = {}
                for i, emotion in enumerate(self.config.NAVARASA_EMOTIONS):
                    emotions_dict[emotion] = float(emotion_probs[i])
                    
                return {
                    "text": text,
                    "dominant_emotion": self.config.NAVARASA_EMOTIONS[dominant_emotion_idx],
                    "dominant_emotion_score": float(emotion_probs[dominant_emotion_idx]),
                    "emotion_intensity": emotion_intensity,
                    "all_emotions": emotions_dict
                }
                
        except Exception as e:
            return {"error": f"Error analyzing sentiment: {str(e)}"}
            
    def get_similar_words(self, word: str, top_k: int = 10) -> Dict:
        """
        Get similar words using the model's embeddings
        
        Args:
            word: Input word
            top_k: Number of similar words to return
            
        Returns:
            Dictionary containing similar words
        """
        if not word.strip():
            return {"error": "Please provide a non-empty word"}
            
        try:
            # Check if word is in vocabulary
            if word not in self.tokenizer.word_to_id:
                return {"error": f"Word '{word}' not found in vocabulary"}
                
            word_id = self.tokenizer.word_to_id[word]
            
            # Get word embedding
            word_embedding = self.model.get_embeddings().weight[word_id]
            
            # Compute similarities
            all_embeddings = self.model.get_embeddings().weight
            similarities = torch.cosine_similarity(
                word_embedding.unsqueeze(0), 
                all_embeddings
            )
            
            # Get top-k most similar words
            top_similarities, top_indices = torch.topk(similarities, top_k + 1)
            
            similar_words = []
            for i in range(1, len(top_indices)):  # Skip the word itself
                idx = top_indices[i].item()
                if idx in self.tokenizer.id_to_word:
                    similar_word = self.tokenizer.id_to_word[idx]
                    similarity_score = top_similarities[i].item()
                    similar_words.append({
                        "word": similar_word,
                        "similarity": similarity_score
                    })
                    
            return {
                "query_word": word,
                "similar_words": similar_words
            }
            
        except Exception as e:
            return {"error": f"Error finding similar words: {str(e)}"}
            
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("🕉️  Sanskrit LLM - Interactive Question-Answering System")
        print("="*60)
        print("Ask questions about Sanskrit literature, philosophy, and culture!")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the system")
        print("  'sentiment <text>' - Analyze sentiment of text")
        print("  'similar <word>' - Find similar words")
        print("  'help' - Show this help message")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("🙏 Ask your question: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit']:
                    print("धन्यवाद! Thank you for using Sanskrit LLM!")
                    break
                    
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'quit' or 'exit' - Exit the system")
                    print("  'sentiment <text>' - Analyze sentiment of text")
                    print("  'similar <word>' - Find similar words")
                    print("  'help' - Show this help message")
                    print("  Or simply ask any question about Sanskrit!\n")
                    continue
                    
                elif user_input.lower().startswith('sentiment '):
                    text = user_input[10:].strip()
                    result = self.analyze_sentiment(text)
                    
                    if 'error' in result:
                        print(f"❌ {result['error']}\n")
                    else:
                        print(f"\n📊 Navarasa Sentiment Analysis:")
                        print(f"Text: {result['text']}")
                        print(f"Dominant Emotion: {result['dominant_emotion']} ({result['dominant_emotion_score']:.3f})")
                        print(f"Emotion Intensity: {result['emotion_intensity']:.3f}")
                        print("\nAll Emotions:")
                        for emotion, score in result['all_emotions'].items():
                            if score > 0.1:  # Only show significant emotions
                                print(f"  {emotion}: {score:.3f}")
                        print()
                        
                elif user_input.lower().startswith('similar '):
                    word = user_input[8:].strip()
                    result = self.get_similar_words(word)
                    
                    if 'error' in result:
                        print(f"❌ {result['error']}\n")
                    else:
                        print(f"\n🔍 Similar words to '{result['query_word']}':")
                        for item in result['similar_words']:
                            print(f"  {item['word']}: {item['similarity']:.3f}")
                        print()
                        
                else:
                    # Regular question
                    print("🤔 Thinking...")
                    result = self.answer_question(user_input)
                    
                    if 'error' in result:
                        print(f"❌ {result['error']}\n")
                    else:
                        print(f"\n💡 Answer ({result['method']}):")
                        print(f"{result['answer']}")
                        
                        if 'retrieved_documents' in result and result['retrieved_documents']:
                            print(f"\n📚 Retrieved Documents:")
                            for i, doc in enumerate(result['retrieved_documents'], 1):
                                print(f"  {i}. {doc['content']} (Score: {doc['score']:.3f})")
                                
                        if 'generation_confidence' in result:
                            print(f"\n🎯 Confidence: {result['generation_confidence']:.3f}")
                            
                        print()
                        
            except KeyboardInterrupt:
                print("\n\nधन्यवाद! Thank you for using Sanskrit LLM!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}\n")


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Sanskrit LLM Question-Answering System')
    parser.add_argument('--model-path', required=True, help='Path to the trained model')
    parser.add_argument('--question', help='Single question to answer')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG')
    parser.add_argument('--max-length', type=int, default=150, help='Maximum answer length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Initialize QA system
    try:
        qa_system = SanskritQASystem(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    if args.interactive:
        # Interactive mode
        qa_system.interactive_mode()
    elif args.question:
        # Single question mode
        result = qa_system.answer_question(
            args.question,
            use_rag=not args.no_rag,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("Question:", result['question'])
            print("Answer:", result['answer'])
            if 'retrieved_documents' in result:
                print(f"Retrieved {len(result['retrieved_documents'])} documents")
    else:
        print("Please specify either --question or --interactive mode")
        print("Use --help for more options")


if __name__ == "__main__":
    main() 