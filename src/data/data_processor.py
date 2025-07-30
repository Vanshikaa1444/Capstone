"""
Data Processing Module for Sanskrit LLM
Handles DCS (Digital Corpus of Sanskrit) data and creates training datasets
"""

import os
import re
import json
import pickle
import requests
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import unicodedata
from tqdm import tqdm

@dataclass
class SanskritText:
    """Data class for Sanskrit text with metadata"""
    text: str
    annotations: Dict[str, Any]
    metadata: Dict[str, Any]
    source: str
    text_id: str

@dataclass
class ProcessedDataset:
    """Data class for processed dataset"""
    texts: List[str]
    questions: List[str]
    answers: List[str]
    contexts: List[str]
    metadata: List[Dict]

class DCSDataProcessor:
    """
    Digital Corpus of Sanskrit (DCS) data processor
    Handles downloading, parsing, and preprocessing of DCS data
    """
    
    def __init__(self, config):
        self.config = config
        self.base_url = config.DCS_BASE_URL
        self.data_dir = Path(config.RAW_DATA_DIR)
        self.processed_dir = Path(config.PROCESSED_DATA_DIR)
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanskrit text patterns
        self._init_sanskrit_patterns()
        
    def _init_sanskrit_patterns(self):
        """Initialize Sanskrit text processing patterns"""
        # Devanagari range
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        
        # Verse patterns
        self.verse_pattern = re.compile(r'\(\d+\.\d+\)')
        
        # Annotation patterns
        self.annotation_pattern = re.compile(r'([a-zA-Z]+\.)+\s*[a-zA-Z]*\.?\s*[a-zA-Z]*\.?')
        
        # Sanskrit punctuation
        self.sanskrit_punct = re.compile(r'[।॥]')
        
        # Common Sanskrit words for filtering
        self.common_words = {
            'च', 'वा', 'तु', 'हि', 'एव', 'अपि', 'किन्तु', 'यदि', 'तदा',
            'सः', 'सा', 'तत्', 'एतत्', 'इदम्', 'यत्', 'किम्', 'कः', 'का'
        }
        
    def download_dcs_texts(self, text_list: List[str] = None) -> List[SanskritText]:
        """
        Download Sanskrit texts from DCS
        
        Args:
            text_list: List of specific text IDs to download
            
        Returns:
            List of SanskritText objects
        """
        sanskrit_texts = []
        
        # Sample DCS text structure (would be replaced with actual API calls)
        sample_texts = [
            {
                "text_id": "mahabharata_01",
                "title": "Mahabharata - Book 1",
                "content": """
granthaprastāvanā
śriyaṃ sarasvatīṃ gaurīṃ gaṇeśaṃ skandamīśvaram / (1.1) Par.?
śrī ac.s.f.
sarasvatī ac.s.f.
gaurī ac.s.f.
gaṇeśa ac.s.m.
skanda ac.s.m.
∞ īśvara ac.s.m.
brahmāṇaṃ vahnimindrādīn vāsudevaṃ namāmyaham // (1.2) Par.?
brahman ac.s.m.
vahni ac.s.m.
∞ indra comp.
∞ ādi ac.p.m.
vāsudeva ac.s.m.
nam 1. sg., Pre. ind.
∞ mad. n.s.a.
                """,
                "metadata": {
                    "genre": "epic",
                    "period": "classical",
                    "author": "vyasa"
                }
            },
            {
                "text_id": "bhagavad_gita_02",
                "title": "Bhagavad Gita - Chapter 2",
                "content": """
sūta uvāca / (4.1) Par.?
sārātsāro hi bhagavān viṣṇuḥ sargādikṛdvibhuḥ / (4.2) Par.?
brahmāhamasmi taṃ jñātvā sarvajñatvaṃ prajāyate // (4.3) Par.?
dve brahmaṇī veditavye śabdabrahma paraṃ ca yat / (5.1) Par.?
dve vidye veditavye hi iti cātharvaṇī śrutiḥ // (5.2) Par.?
                """,
                "metadata": {
                    "genre": "philosophical",
                    "period": "classical",
                    "author": "vyasa"
                }
            }
        ]
        
        for text_data in sample_texts:
            sanskrit_text = SanskritText(
                text=text_data["content"],
                annotations=self._parse_annotations(text_data["content"]),
                metadata=text_data["metadata"],
                source="dcs",
                text_id=text_data["text_id"]
            )
            sanskrit_texts.append(sanskrit_text)
            
        print(f"Downloaded {len(sanskrit_texts)} Sanskrit texts")
        return sanskrit_texts
        
    def _parse_annotations(self, text: str) -> Dict[str, Any]:
        """Parse DCS annotations from text"""
        annotations = {
            "morphological": [],
            "syntactic": [],
            "semantic": []
        }
        
        lines = text.strip().split('\n')
        current_verse = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if it's a verse line
            if self.verse_pattern.search(line):
                current_verse = line
                continue
                
            # Check if it's an annotation line
            if self.annotation_pattern.search(line):
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    annotation = parts[1]
                    
                    annotations["morphological"].append({
                        "word": word,
                        "annotation": annotation,
                        "verse": current_verse
                    })
                    
        return annotations
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize Sanskrit text"""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove annotation lines (keep only main text)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Keep lines that contain Devanagari or transliterated Sanskrit
            if (self.devanagari_pattern.search(line) or 
                self.verse_pattern.search(line) or
                any(char.isalpha() and ord(char) < 128 for char in line)):
                
                # Remove annotation markers
                line = re.sub(r'\([^)]*\)', '', line)  # Remove (1.1) markers
                line = re.sub(r'Par\.\?', '', line)   # Remove Par.? markers
                line = re.sub(r'/\s*$', '', line)     # Remove trailing /
                line = re.sub(r'//\s*$', '', line)    # Remove trailing //
                
                # Clean up whitespace
                line = re.sub(r'\s+', ' ', line).strip()
                
                if line and not self.annotation_pattern.match(line):
                    cleaned_lines.append(line)
                    
        cleaned_text = ' '.join(cleaned_lines)
        
        # Additional cleaning
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
        
    def create_qa_pairs(self, sanskrit_texts: List[SanskritText]) -> List[Dict]:
        """
        Create question-answer pairs from Sanskrit texts
        
        Args:
            sanskrit_texts: List of SanskritText objects
            
        Returns:
            List of QA pairs
        """
        qa_pairs = []
        
        for text_obj in sanskrit_texts:
            text = self.clean_text(text_obj.text)
            
            # Split text into sentences/verses
            sentences = self._split_into_verses(text)
            
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) < 3:  # Skip very short sentences
                    continue
                    
                # Generate different types of questions
                qa_pairs.extend(self._generate_comprehension_questions(sentence, text_obj.metadata))
                qa_pairs.extend(self._generate_contextual_questions(sentence, sentences, i, text_obj.metadata))
                qa_pairs.extend(self._generate_analysis_questions(sentence, text_obj.metadata))
                
        print(f"Generated {len(qa_pairs)} QA pairs")
        return qa_pairs
        
    def _split_into_verses(self, text: str) -> List[str]:
        """Split text into verses or meaningful chunks"""
        # Split by Sanskrit punctuation
        verses = re.split(r'[।॥]', text)
        
        # Also split by / and // (common DCS separators)
        all_verses = []
        for verse in verses:
            sub_verses = re.split(r'\s*/\s*|\s*//\s*', verse)
            all_verses.extend(sub_verses)
            
        # Clean and filter verses
        cleaned_verses = []
        for verse in all_verses:
            verse = verse.strip()
            if verse and len(verse.split()) >= 3:
                cleaned_verses.append(verse)
                
        return cleaned_verses
        
    def _generate_comprehension_questions(self, sentence: str, metadata: Dict) -> List[Dict]:
        """Generate comprehension questions for a sentence"""
        questions = []
        
        # Question templates
        templates = [
            "इस श्लोक का अर्थ क्या है?",
            "यहाँ क्या कहा गया है?",
            "इस वाक्य में मुख्य संदेश क्या है?",
            "What is the meaning of this verse?",
            "What is being said here?",
            "What is the main message in this sentence?"
        ]
        
        for template in templates[:2]:  # Limit to avoid too many questions
            qa_pair = {
                "question": template,
                "answer": sentence,
                "context": sentence,
                "question_type": "comprehension",
                "metadata": metadata.copy()
            }
            questions.append(qa_pair)
            
        return questions
        
    def _generate_contextual_questions(self, sentence: str, all_sentences: List[str], 
                                     current_idx: int, metadata: Dict) -> List[Dict]:
        """Generate contextual questions using surrounding sentences"""
        questions = []
        
        # Create context from surrounding sentences
        context_start = max(0, current_idx - 2)
        context_end = min(len(all_sentences), current_idx + 3)
        context = ' '.join(all_sentences[context_start:context_end])
        
        context_templates = [
            "इस संदर्भ में यह श्लोक क्या बताता है?",
            "पूर्व और परवर्ती श्लोकों के साथ यह कैसे जुड़ता है?",
            "In this context, what does this verse convey?",
            "How does this connect with the preceding and following verses?"
        ]
        
        for template in context_templates[:2]:
            qa_pair = {
                "question": template,
                "answer": sentence,
                "context": context,
                "question_type": "contextual",
                "metadata": metadata.copy()
            }
            questions.append(qa_pair)
            
        return questions
        
    def _generate_analysis_questions(self, sentence: str, metadata: Dict) -> List[Dict]:
        """Generate analytical questions about the sentence"""
        questions = []
        
        # Genre-specific questions
        if metadata.get("genre") == "philosophical":
            templates = [
                "इस श्लोक में कौन सा दार्शनिक सिद्धांत प्रतिपादित है?",
                "यह श्लोक जीवन के बारे में क्या शिक्षा देता है?",
                "What philosophical principle is expounded in this verse?",
                "What lesson about life does this verse teach?"
            ]
        elif metadata.get("genre") == "epic":
            templates = [
                "इस श्लोक में कौन सा पात्र या घटना वर्णित है?",
                "यह कथा के मुख्य संदेश में कैसे योगदान देता है?",
                "Which character or event is described in this verse?",
                "How does this contribute to the main message of the story?"
            ]
        else:
            templates = [
                "इस श्लोक का साहित्यिक महत्व क्या है?",
                "यह श्लोक किस भाव या रस को व्यक्त करता है?",
                "What is the literary significance of this verse?",
                "What emotion or rasa does this verse express?"
            ]
            
        for template in templates[:1]:
            qa_pair = {
                "question": template,
                "answer": sentence,
                "context": sentence,
                "question_type": "analytical",
                "metadata": metadata.copy()
            }
            questions.append(qa_pair)
            
        return questions
        
    def create_sentiment_dataset(self, sanskrit_texts: List[SanskritText]) -> List[Dict]:
        """Create dataset for Navarasa sentiment analysis"""
        sentiment_data = []
        
        # Emotion keywords for labeling (simplified approach)
        emotion_keywords = {
            0: ["प्रेम", "प्रीति", "काम", "सुन्दर", "रूप", "कान्त"],  # shringara
            1: ["हास", "विनोद", "हास्य", "चपल", "कौतुक"],  # hasya
            2: ["करुणा", "दुःख", "शोक", "विलाप", "क्रन्दन"],  # karuna
            3: ["क्रोध", "कोप", "रौद्र", "मन्यु", "अमर्ष"],  # raudra
            4: ["वीर", "साहस", "शौर्य", "युद्ध", "विजय"],  # veera
            5: ["भय", "त्रास", "आतंक", "भीत", "त्रस्त"],  # bhayanaka
            6: ["घृणा", "जुगुप्सा", "बिभत्स", "अरुचि"],  # bibhatsa
            7: ["अद्भुत", "आश्चर्य", "विस्मय", "चमत्कार"],  # adbhuta
            8: ["शान्त", "निर्वाण", "मोक्ष", "समाधि", "प्रशान्त"]  # shanta
        }
        
        for text_obj in sanskrit_texts:
            sentences = self._split_into_verses(self.clean_text(text_obj.text))
            
            for sentence in sentences:
                if len(sentence.split()) < 5:
                    continue
                    
                # Simple keyword-based emotion labeling
                emotion_scores = [0] * 9
                for emotion_idx, keywords in emotion_keywords.items():
                    for keyword in keywords:
                        if keyword in sentence:
                            emotion_scores[emotion_idx] += 1
                            
                # Assign primary emotion
                primary_emotion = np.argmax(emotion_scores) if max(emotion_scores) > 0 else 8  # Default to shanta
                
                sentiment_item = {
                    "text": sentence,
                    "emotion": primary_emotion,
                    "emotion_scores": emotion_scores,
                    "metadata": text_obj.metadata.copy()
                }
                sentiment_data.append(sentiment_item)
                
        print(f"Created {len(sentiment_data)} sentiment samples")
        return sentiment_data
        
    def save_processed_data(self, data: Any, filename: str):
        """Save processed data to file"""
        filepath = self.processed_dir / filename
        
        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif filename.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            
        print(f"Saved processed data to {filepath}")
        
    def load_processed_data(self, filename: str) -> Any:
        """Load processed data from file"""
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")


class SanskritDatasetBuilder:
    """Build complete training datasets for Sanskrit LLM"""
    
    def __init__(self, config):
        self.config = config
        self.processor = DCSDataProcessor(config)
        
    def build_complete_dataset(self) -> ProcessedDataset:
        """Build complete dataset with all components"""
        print("Building complete Sanskrit LLM dataset...")
        
        # Download and process DCS texts
        sanskrit_texts = self.processor.download_dcs_texts()
        
        # Create QA pairs
        qa_pairs = self.processor.create_qa_pairs(sanskrit_texts)
        
        # Create sentiment dataset
        sentiment_data = self.processor.create_sentiment_dataset(sanskrit_texts)
        
        # Extract components
        texts = [self.processor.clean_text(text.text) for text in sanskrit_texts]
        questions = [qa['question'] for qa in qa_pairs]
        answers = [qa['answer'] for qa in qa_pairs]
        contexts = [qa['context'] for qa in qa_pairs]
        metadata = [qa['metadata'] for qa in qa_pairs]
        
        dataset = ProcessedDataset(
            texts=texts,
            questions=questions,
            answers=answers,
            contexts=contexts,
            metadata=metadata
        )
        
        # Save all datasets
        self.processor.save_processed_data(qa_pairs, 'qa_pairs.json')
        self.processor.save_processed_data(sentiment_data, 'sentiment_data.json')
        self.processor.save_processed_data(texts, 'cleaned_texts.json')
        
        print("Dataset building complete!")
        return dataset
        
    def create_train_val_test_split(self, dataset: ProcessedDataset, 
                                  train_ratio: float = 0.8,
                                  val_ratio: float = 0.1,
                                  test_ratio: float = 0.1) -> Dict[str, ProcessedDataset]:
        """Split dataset into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Combine all data
        all_data = list(zip(dataset.questions, dataset.answers, dataset.contexts, dataset.metadata))
        
        # Shuffle data
        random.shuffle(all_data)
        
        # Calculate split indices
        n_total = len(all_data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split data
        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]
        
        # Create dataset objects
        splits = {}
        for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if data:
                questions, answers, contexts, metadata = zip(*data)
                splits[name] = ProcessedDataset(
                    texts=dataset.texts,  # Keep all texts for context
                    questions=list(questions),
                    answers=list(answers),
                    contexts=list(contexts),
                    metadata=list(metadata)
                )
            else:
                splits[name] = ProcessedDataset([], [], [], [], [])
                
        print(f"Dataset split: Train={len(splits['train'].questions)}, "
              f"Val={len(splits['val'].questions)}, Test={len(splits['test'].questions)}")
              
        return splits 