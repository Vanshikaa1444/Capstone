"""
Sanskrit Tokenizer for handling complex morphological structure
Supports Devanagari script, morphological analysis, and DCS annotations
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
import pickle

class SanskritTokenizer:
    def __init__(self, vocab_size: int = 50000, special_tokens: Dict[str, str] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]"
        }
        
        # Vocabulary mappings
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_freq = defaultdict(int)
        
        # Sanskrit specific patterns
        self._init_sanskrit_patterns()
        
        # Morphological analysis patterns
        self._init_morphological_patterns()
        
        # Initialize with special tokens
        self._add_special_tokens()
        
    def _init_sanskrit_patterns(self):
        """Initialize Sanskrit-specific regex patterns"""
        # Devanagari character ranges
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        
        # Vowel patterns
        self.vowel_pattern = re.compile(r'[अआइईउऊऋएऐओऔ]')
        self.vowel_marks = re.compile(r'[\u093E-\u094C\u0962\u0963]')
        
        # Consonant patterns
        self.consonant_pattern = re.compile(r'[क-ह]')
        
        # Virama (halant) pattern
        self.virama_pattern = re.compile(r'\u094D')
        
        # Sanskrit punctuation
        self.punct_pattern = re.compile(r'[।॥\.\,\;\:\!\?\/\(\)\[\]\{\}]')
        
        # Number patterns
        self.number_pattern = re.compile(r'[\u0966-\u096F]+')
        
        # Sandhi junction patterns (simplified)
        self.sandhi_patterns = [
            (r'अ\s*\+\s*अ', r'आ'),
            (r'आ\s*\+\s*अ', r'आ'),
            (r'इ\s*\+\s*अ', r'य'),
            (r'उ\s*\+\s*अ', r'व'),
            (r'ऋ\s*\+\s*अ', r'अर्'),
        ]
        
    def _init_morphological_patterns(self):
        """Initialize patterns for morphological analysis"""
        # Case endings for nouns
        self.case_endings = {
            'nominative': ['ः', 'ाः', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ'],
            'accusative': ['म्', 'ान्', 'ीन्', 'ून्', 'ृन्'],
            'instrumental': ['ेन', 'या', 'ैः'],
            'dative': ['ाय', 'ै', 'े'],
            'ablative': ['ात्', 'स्मात्', 'ेभ्यः'],
            'genitive': ['स्य', 'या', 'ाणाम्'],
            'locative': ['े', 'ि', 'ौ', 'ेषु']
        }
        
        # Verb endings
        self.verb_endings = {
            'present': ['ति', 'तः', 'न्ति', 'सि', 'थः', 'थ', 'मि', 'वः', 'मः'],
            'past': ['त्', 'ताम्', 'न्', 'स्', 'तम्', 'त', 'म्', 'व', 'म'],
            'future': ['ष्यति', 'ष्यतः', 'ष्यन्ति', 'ष्यसि', 'ष्यथः', 'ष्यथ']
        }
        
        # Morphological tags mapping
        self.morph_tags = {
            'n.': 'noun',
            'v.': 'verb',
            'adj.': 'adjective',
            'adv.': 'adverb',
            'pron.': 'pronoun',
            'prep.': 'preposition',
            'conj.': 'conjunction',
            'interj.': 'interjection',
            's.': 'singular',
            'p.': 'plural',
            'd.': 'dual',
            'm.': 'masculine',
            'f.': 'feminine',
            'n.': 'neuter',
            'ac.': 'accusative',
            'no.': 'nominative',
            'in.': 'instrumental',
            'da.': 'dative',
            'ab.': 'ablative',
            'ge.': 'genitive',
            'lo.': 'locative',
            'vo.': 'vocative'
        }
        
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        for i, token in enumerate(self.special_tokens.values()):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            
    def normalize_text(self, text: str) -> str:
        """Normalize Sanskrit text"""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common transliteration issues
        text = text.replace('|', '।')  # Replace pipe with danda
        text = text.replace('||', '॥')  # Replace double pipe with double danda
        
        return text.strip()
        
    def handle_sandhi(self, text: str) -> str:
        """Apply basic sandhi resolution"""
        if not hasattr(self, 'handle_sandhi_enabled') or not self.handle_sandhi_enabled:
            return text
            
        for pattern, replacement in self.sandhi_patterns:
            text = re.sub(pattern, replacement, text)
            
        return text
        
    def parse_dcs_annotation(self, line: str) -> Tuple[str, Dict]:
        """Parse DCS annotation format"""
        parts = line.strip().split()
        if len(parts) < 2:
            return line.strip(), {}
            
        word = parts[0]
        annotations = {}
        
        # Parse morphological information
        for part in parts[1:]:
            if '.' in part:
                # This is likely a morphological tag
                tag_parts = part.split('.')
                for tag_part in tag_parts:
                    if tag_part in self.morph_tags:
                        category = self.morph_tags[tag_part]
                        if 'morphology' not in annotations:
                            annotations['morphology'] = []
                        annotations['morphology'].append(category)
                        
        return word, annotations
        
    def extract_root_word(self, word: str, annotations: Dict = None) -> str:
        """Extract root form of Sanskrit word"""
        # Simple stemming based on common endings
        for case_type, endings in self.case_endings.items():
            for ending in endings:
                if word.endswith(ending) and len(word) > len(ending):
                    return word[:-len(ending)]
                    
        for verb_type, endings in self.verb_endings.items():
            for ending in endings:
                if word.endswith(ending) and len(word) > len(ending):
                    return word[:-len(ending)]
                    
        return word
        
    def tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single Sanskrit word into subwords"""
        # Remove punctuation
        word = re.sub(self.punct_pattern, '', word)
        
        if len(word) == 0:
            return []
            
        # For Sanskrit, we'll use character-level tokenization for now
        # This can be enhanced with morpheme-level tokenization
        tokens = []
        
        # Split on virama (halant) to separate consonant clusters
        parts = word.split('\u094D')
        
        for i, part in enumerate(parts):
            if i > 0:
                tokens.append('\u094D')  # Add virama back
            if part:
                # Further split into characters
                tokens.extend(list(part))
                
        return [token for token in tokens if token.strip()]
        
    def tokenize_sentence(self, sentence: str) -> List[str]:
        """Tokenize a Sanskrit sentence"""
        # Normalize text
        sentence = self.normalize_text(sentence)
        
        # Handle sandhi
        sentence = self.handle_sandhi(sentence)
        
        # Split by whitespace and punctuation
        words = re.findall(r'\S+', sentence)
        
        tokens = []
        for word in words:
            if re.match(self.punct_pattern, word):
                tokens.append(word)
            else:
                word_tokens = self.tokenize_word(word)
                tokens.extend(word_tokens)
                
        return tokens
        
    def process_dcs_text(self, text: str) -> List[Tuple[str, Dict]]:
        """Process DCS format text with annotations"""
        lines = text.strip().split('\n')
        processed_tokens = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check if line contains annotations
            if any(tag in line for tag in self.morph_tags.keys()):
                word, annotations = self.parse_dcs_annotation(line)
                if word:
                    processed_tokens.append((word, annotations))
            else:
                # Regular text line
                tokens = self.tokenize_sentence(line)
                for token in tokens:
                    processed_tokens.append((token, {}))
                    
        return processed_tokens
        
    def build_vocabulary(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from training texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            if text.strip():
                processed_tokens = self.process_dcs_text(text)
                for token, _ in processed_tokens:
                    self.word_freq[token] += 1
                    
        # Sort by frequency and select top words
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add words to vocabulary
        current_id = len(self.special_tokens)
        
        for word, freq in sorted_words:
            if freq >= min_freq and current_id < self.vocab_size:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_id
                    self.id_to_word[current_id] = word
                    current_id += 1
                    
        print(f"Vocabulary built with {len(self.word_to_id)} tokens")
        
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        processed_tokens = self.process_dcs_text(text)
        
        # Convert tokens to IDs
        token_ids = [self.word_to_id.get(self.special_tokens["bos_token"], 0)]
        
        for token, _ in processed_tokens:
            if len(token_ids) >= max_length - 1:  # Leave space for EOS
                break
            token_id = self.word_to_id.get(token, self.word_to_id.get(self.special_tokens["unk_token"], 1))
            token_ids.append(token_id)
            
        # Add EOS token
        token_ids.append(self.word_to_id.get(self.special_tokens["eos_token"], 2))
        
        # Pad if necessary
        while len(token_ids) < max_length:
            token_ids.append(self.word_to_id.get(self.special_tokens["pad_token"], 0))
            
        return token_ids[:max_length]
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if token not in [self.special_tokens["pad_token"], 
                               self.special_tokens["bos_token"], 
                               self.special_tokens["eos_token"]]:
                    tokens.append(token)
                    
        return ''.join(tokens)
        
    def save(self, filepath: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'word_freq': dict(self.word_freq),
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
            
        print(f"Tokenizer saved to {filepath}")
        
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
            
        self.word_to_id = tokenizer_data['word_to_id']
        self.id_to_word = tokenizer_data['id_to_word']
        self.word_freq = defaultdict(int, tokenizer_data['word_freq'])
        self.special_tokens = tokenizer_data['special_tokens']
        self.vocab_size = tokenizer_data['vocab_size']
        
        print(f"Tokenizer loaded from {filepath}")
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_to_id)
        
    def get_token_id(self, token: str) -> int:
        """Get token ID for a given token"""
        return self.word_to_id.get(token, self.word_to_id.get(self.special_tokens["unk_token"], 1)) 