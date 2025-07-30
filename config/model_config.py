"""
Configuration file for Sanskrit LLM
Contains all model parameters, training settings, and hyperparameters
"""

class SanskritLLMConfig:
    # Model Architecture Parameters
    MODEL_NAME = "sanskrit_llm"
    VOCAB_SIZE = 50000
    MAX_SEQ_LENGTH = 512
    
    # Transformer Architecture
    NUM_LAYERS = 8
    NUM_ATTENTION_HEADS = 8
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 2048
    DROPOUT_RATE = 0.1
    ATTENTION_DROPOUT = 0.1
    
    # Embedding Parameters
    EMBEDDING_DIM = 512
    WORD2VEC_WINDOW = 5
    WORD2VEC_MIN_COUNT = 2
    WORD2VEC_EPOCHS = 100
    WORD2VEC_VECTOR_SIZE = 300
    
    # Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 10000
    MAX_EPOCHS = 100
    GRADIENT_CLIP_NORM = 1.0
    
    # Optimizer Settings
    OPTIMIZER = "adam"
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8
    
    # Loss Function
    LABEL_SMOOTHING = 0.1
    
    # Tokenizer Settings
    TOKENIZER_MODEL_TYPE = "sanskrit_wordpiece"
    SPECIAL_TOKENS = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]"
    }
    
    # Data Processing
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 1000
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # RAG Parameters
    RAG_TOP_K = 5
    RAG_CHUNK_SIZE = 256
    RAG_OVERLAP = 50
    RETRIEVAL_THRESHOLD = 0.7
    
    # Navarasa Sentiment Classes
    NAVARASA_EMOTIONS = [
        "shringara",  # Love/Romance
        "hasya",      # Laughter/Comedy
        "karuna",     # Compassion/Sadness
        "raudra",     # Anger
        "veera",      # Heroism/Courage
        "bhayanaka",  # Fear/Terror
        "bibhatsa",   # Disgust
        "adbhuta",    # Wonder/Amazement
        "shanta"      # Peace/Tranquility
    ]
    
    # Evaluation Parameters
    BLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
    EVAL_BATCH_SIZE = 16
    
    # File Paths
    DATA_DIR = "data/"
    RAW_DATA_DIR = "data/raw/"
    PROCESSED_DATA_DIR = "data/processed/"
    MODEL_SAVE_DIR = "models/"
    LOG_DIR = "logs/"
    
    # DCS Data Parameters
    DCS_BASE_URL = "http://www.sanskrit-linguistics.org/dcs/"
    ANNOTATION_LEVELS = ["word", "morphology", "syntax"]
    
    # Device Settings
    DEVICE = "cuda"  # Will be set to "cpu" if CUDA is not available
    
    # Logging
    LOG_LEVEL = "INFO"
    SAVE_EVERY_N_STEPS = 1000
    EVAL_EVERY_N_STEPS = 5000
    
    # QA System Parameters
    MAX_ANSWER_LENGTH = 150
    MIN_ANSWER_LENGTH = 5
    CONTEXT_WINDOW = 3
    
    # Sanskrit Specific Settings
    DEVANAGARI_UNICODE_RANGE = (0x0900, 0x097F)
    NORMALIZATION_FORM = "NFC"  # Unicode normalization
    HANDLE_SANDHI = True
    USE_STEMMING = True 