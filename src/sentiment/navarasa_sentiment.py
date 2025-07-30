"""
Navarasa Sentiment Analysis Layer
Enhances understanding of Sanskrit poetry, literature and philosophy
to produce emotion-aware answers based on the nine rasas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

class NavaraSentimentLayer(nn.Module):
    """
    Navarasa Sentiment Analysis Layer for Sanskrit Literature
    Recognizes and processes the nine fundamental emotions (rasas) in Sanskrit texts
    """
    
    def __init__(self, config):
        super(NavaraSentimentLayer, self).__init__()
        
        self.config = config
        self.d_model = config.HIDDEN_SIZE
        self.num_emotions = len(config.NAVARASA_EMOTIONS)
        self.emotion_names = config.NAVARASA_EMOTIONS
        
        # Emotion classification layers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, self.num_emotions)
        )
        
        # Emotion-aware attention mechanism
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Emotion embedding layer
        self.emotion_embeddings = nn.Embedding(self.num_emotions, self.d_model)
        
        # Emotion intensity predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Multi-emotion fusion layer
        self.emotion_fusion = nn.Linear(self.d_model * self.num_emotions, self.d_model)
        
        # Sanskrit literature context embeddings
        self.register_rasa_patterns()
        
    def register_rasa_patterns(self):
        """Register Sanskrit rasa patterns and keywords"""
        self.rasa_patterns = {
            'shringara': [  # Love/Romance
                'प्रेम', 'प्रीति', 'काम', 'मदन', 'स्नेह', 'अनुराग', 'वल्लभ',
                'प्रिय', 'दयित', 'कान्त', 'सुन्दर', 'रूप', 'लावण्य'
            ],
            'hasya': [  # Laughter/Comedy
                'हास', 'विनोद', 'उपहास', 'परिहास', 'मजाक', 'कौतुक',
                'विकट', 'वक्र', 'चपल', 'कुटिल'
            ],
            'karuna': [  # Compassion/Sadness
                'करुणा', 'दुःख', 'शोक', 'विलाप', 'आर्त', 'क्रन्दन',
                'अन्तक', 'मृत्यु', 'वियोग', 'दीन', 'खिन्न'
            ],
            'raudra': [  # Anger
                'क्रोध', 'कोप', 'रौद्र', 'मन्यु', 'अमर्ष', 'रोष',
                'क्षुब्ध', 'कुपित', 'तप्त', 'ज्वल'
            ],
            'veera': [  # Heroism/Courage
                'वीर', 'साहस', 'शौर्य', 'पराक्रम', 'धैर्य', 'उत्साह',
                'युद्ध', 'रण', 'विजय', 'जय', 'बल', 'शक्ति'
            ],
            'bhayanaka': [  # Fear/Terror
                'भय', 'त्रास', 'आतंक', 'डर', 'घबराहट', 'विकम्प',
                'भीत', 'त्रस्त', 'आश्चर्य', 'विस्मय'
            ],
            'bibhatsa': [  # Disgust
                'बिभत्स', 'घृणा', 'जुगुप्सा', 'अरुचि', 'वितृष्णा',
                'गन्ध', 'दुर्गन्ध', 'कलुष', 'मलिन'
            ],
            'adbhuta': [  # Wonder/Amazement
                'अद्भुत', 'आश्चर्य', 'विस्मय', 'चमत्कार', 'महत्',
                'अलौकिक', 'दिव्य', 'अपूर्व', 'नव', 'विलक्षण'
            ],
            'shanta': [  # Peace/Tranquility
                'शान्त', 'निर्वाण', 'मोक्ष', 'समाधि', 'स्थिर', 'धीर',
                'प्रशान्त', 'निर्मल', 'पवित्र', 'शुद्ध', 'निष्काम'
            ]
        }
        
        # Create pattern embeddings
        self.pattern_embeddings = {}
        for emotion_idx, (emotion, patterns) in enumerate(self.rasa_patterns.items()):
            self.pattern_embeddings[emotion] = emotion_idx
            
    def detect_rasa_patterns(self, text_tokens: List[str]) -> Dict[str, float]:
        """Detect rasa patterns in Sanskrit text"""
        emotion_scores = defaultdict(float)
        
        for emotion, patterns in self.rasa_patterns.items():
            for token in text_tokens:
                if token in patterns:
                    emotion_scores[emotion] += 1.0
                    
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
                
        return dict(emotion_scores)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                text_tokens: Optional[List[List[str]]] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Navarasa sentiment layer
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_length, d_model]
            text_tokens: Original text tokens for pattern matching
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing sentiment outputs
        """
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device
        
        outputs = {}
        
        # 1. Emotion Classification
        # Pool hidden states for sequence-level emotion classification
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, d_model]
        emotion_logits = self.emotion_classifier(pooled_hidden)  # [batch_size, num_emotions]
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        outputs['emotion_logits'] = emotion_logits
        outputs['emotion_probs'] = emotion_probs
        outputs['dominant_emotion'] = torch.argmax(emotion_probs, dim=-1)
        
        # 2. Emotion Intensity Prediction
        intensity_scores = self.intensity_predictor(pooled_hidden)  # [batch_size, 1]
        outputs['emotion_intensity'] = intensity_scores
        
        # 3. Pattern-based Emotion Detection (if text tokens provided)
        if text_tokens is not None:
            pattern_emotions = []
            for tokens in text_tokens:
                pattern_scores = self.detect_rasa_patterns(tokens)
                pattern_vector = torch.zeros(self.num_emotions, device=device)
                
                for emotion, score in pattern_scores.items():
                    if emotion in self.emotion_names:
                        emotion_idx = self.emotion_names.index(emotion)
                        pattern_vector[emotion_idx] = score
                        
                pattern_emotions.append(pattern_vector)
                
            pattern_emotions = torch.stack(pattern_emotions)  # [batch_size, num_emotions]
            outputs['pattern_emotions'] = pattern_emotions
            
            # Combine neural and pattern-based predictions
            combined_emotions = 0.7 * emotion_probs + 0.3 * pattern_emotions
            outputs['combined_emotions'] = combined_emotions
        
        # 4. Emotion-aware Attention
        # Create emotion queries using dominant emotions
        dominant_emotions = torch.argmax(emotion_probs, dim=-1)
        emotion_queries = self.emotion_embeddings(dominant_emotions).unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Apply emotion-aware attention
        emotion_attended, attention_weights = self.emotion_attention(
            emotion_queries, hidden_states, hidden_states
        )
        
        outputs['emotion_attended'] = emotion_attended.squeeze(1)  # [batch_size, d_model]
        
        if return_attention:
            outputs['emotion_attention_weights'] = attention_weights
            
        # 5. Multi-emotion Fusion
        # Create representations for all emotions
        all_emotion_embeds = self.emotion_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply attention with each emotion
        emotion_representations = []
        for i in range(self.num_emotions):
            emotion_query = all_emotion_embeds[:, i:i+1, :]  # [batch_size, 1, d_model]
            emotion_rep, _ = self.emotion_attention(emotion_query, hidden_states, hidden_states)
            emotion_representations.append(emotion_rep.squeeze(1))
            
        # Concatenate all emotion representations
        fused_emotions = torch.cat(emotion_representations, dim=-1)  # [batch_size, d_model * num_emotions]
        
        # Project back to d_model size
        emotion_enhanced = self.emotion_fusion(fused_emotions)  # [batch_size, d_model]
        outputs['emotion_enhanced'] = emotion_enhanced
        
        # 6. Emotion-specific Answer Generation Bias
        # Create emotion-specific biases for different types of responses
        emotion_biases = self.create_emotion_biases(emotion_probs)
        outputs['emotion_biases'] = emotion_biases
        
        return outputs
        
    def create_emotion_biases(self, emotion_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create emotion-specific biases for answer generation"""
        biases = {}
        
        # Shringara (Love) - bias towards romantic, beautiful language
        shringara_weight = emotion_probs[:, 0].unsqueeze(-1)
        biases['beauty_bias'] = shringara_weight * 2.0
        
        # Hasya (Comedy) - bias towards lighter, humorous responses
        hasya_weight = emotion_probs[:, 1].unsqueeze(-1)
        biases['humor_bias'] = hasya_weight * 1.5
        
        # Karuna (Compassion) - bias towards empathetic responses
        karuna_weight = emotion_probs[:, 2].unsqueeze(-1)
        biases['compassion_bias'] = karuna_weight * 2.5
        
        # Raudra (Anger) - bias towards direct, forceful language
        raudra_weight = emotion_probs[:, 3].unsqueeze(-1)
        biases['intensity_bias'] = raudra_weight * 1.8
        
        # Veera (Heroism) - bias towards inspirational language
        veera_weight = emotion_probs[:, 4].unsqueeze(-1)
        biases['inspiration_bias'] = veera_weight * 2.2
        
        # Bhayanaka (Fear) - bias towards careful, cautious responses
        bhayanaka_weight = emotion_probs[:, 5].unsqueeze(-1)
        biases['caution_bias'] = bhayanaka_weight * 1.3
        
        # Bibhatsa (Disgust) - bias towards avoiding certain topics
        bibhatsa_weight = emotion_probs[:, 6].unsqueeze(-1)
        biases['avoidance_bias'] = bibhatsa_weight * 0.5
        
        # Adbhuta (Wonder) - bias towards elaborate, detailed responses
        adbhuta_weight = emotion_probs[:, 7].unsqueeze(-1)
        biases['elaboration_bias'] = adbhuta_weight * 2.0
        
        # Shanta (Peace) - bias towards balanced, philosophical responses
        shanta_weight = emotion_probs[:, 8].unsqueeze(-1)
        biases['wisdom_bias'] = shanta_weight * 2.8
        
        return biases
        
    def get_emotion_name(self, emotion_idx: int) -> str:
        """Get emotion name from index"""
        if 0 <= emotion_idx < len(self.emotion_names):
            return self.emotion_names[emotion_idx]
        return "unknown"
        
    def interpret_emotions(self, emotion_probs: torch.Tensor, threshold: float = 0.1) -> List[Dict]:
        """Interpret emotion probabilities for human understanding"""
        batch_size = emotion_probs.size(0)
        interpretations = []
        
        for i in range(batch_size):
            probs = emotion_probs[i].cpu().numpy()
            interpretation = {
                'dominant_emotion': self.emotion_names[np.argmax(probs)],
                'dominant_score': float(np.max(probs)),
                'all_emotions': {}
            }
            
            for j, emotion in enumerate(self.emotion_names):
                if probs[j] >= threshold:
                    interpretation['all_emotions'][emotion] = float(probs[j])
                    
            interpretations.append(interpretation)
            
        return interpretations


class EmotionAwareResponseGenerator(nn.Module):
    """
    Emotion-Aware Response Generator
    Uses Navarasa sentiment to modify response generation
    """
    
    def __init__(self, config, vocab_size: int):
        super(EmotionAwareResponseGenerator, self).__init__()
        
        self.config = config
        self.d_model = config.HIDDEN_SIZE
        self.vocab_size = vocab_size
        self.num_emotions = len(config.NAVARASA_EMOTIONS)
        
        # Navarasa sentiment layer
        self.sentiment_layer = NavaraSentimentLayer(config)
        
        # Emotion-conditioned vocabulary projection
        self.emotion_vocab_projection = nn.ModuleList([
            nn.Linear(self.d_model, self.vocab_size) for _ in range(self.num_emotions)
        ])
        
        # Base vocabulary projection
        self.base_vocab_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Emotion mixing weights
        self.emotion_mixer = nn.Linear(self.num_emotions, self.num_emotions)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                text_tokens: Optional[List[List[str]]] = None) -> Dict[str, torch.Tensor]:
        """
        Generate emotion-aware responses
        
        Args:
            hidden_states: Hidden states from transformer [batch_size, seq_length, d_model]
            text_tokens: Original text tokens for pattern matching
            
        Returns:
            Dictionary containing emotion-aware outputs
        """
        # Get sentiment analysis
        sentiment_outputs = self.sentiment_layer(hidden_states, text_tokens)
        
        # Base vocabulary logits
        base_logits = self.base_vocab_projection(hidden_states)
        
        # Emotion-specific vocabulary logits
        emotion_probs = sentiment_outputs['emotion_probs']  # [batch_size, num_emotions]
        
        emotion_logits_list = []
        for i in range(self.num_emotions):
            emotion_logits = self.emotion_vocab_projection[i](hidden_states)
            emotion_logits_list.append(emotion_logits)
            
        emotion_logits = torch.stack(emotion_logits_list, dim=1)  # [batch_size, num_emotions, seq_length, vocab_size]
        
        # Mix emotion-specific logits based on emotion probabilities
        emotion_weights = self.emotion_mixer(emotion_probs)  # [batch_size, num_emotions]
        emotion_weights = F.softmax(emotion_weights, dim=-1)
        
        # Weighted combination of emotion-specific logits
        weighted_emotion_logits = torch.einsum('be,besv->bsv', emotion_weights, emotion_logits)
        
        # Combine base and emotion-aware logits
        final_logits = 0.3 * base_logits + 0.7 * weighted_emotion_logits
        
        # Apply emotion biases
        emotion_biases = sentiment_outputs['emotion_biases']
        for bias_name, bias_value in emotion_biases.items():
            if bias_name == 'elaboration_bias':
                # Encourage longer responses for wonder/amazement
                final_logits[:, :, :] += bias_value.unsqueeze(1) * 0.1
                
        return {
            'emotion_aware_logits': final_logits,
            'base_logits': base_logits,
            'emotion_logits': weighted_emotion_logits,
            'emotion_weights': emotion_weights,
            **sentiment_outputs
        } 