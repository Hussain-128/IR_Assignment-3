"""
Configuration file for the Information Retrieval System
"""
import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_DIR = os.path.join(BASE_DIR, 'index')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Preprocessing parameters
USE_STEMMING = True
USE_LEMMATIZATION = False
REMOVE_STOPWORDS = True
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 20

# Retrieval parameters
TOP_K_RESULTS = 10
USE_TF_IDF = True
USE_BM25 = False

# BM25 parameters (if used)
BM25_K1 = 1.5
BM25_B = 0.75

# TF-IDF parameters
TF_IDF_NORM = 'l2'  # 'l1', 'l2', or None
TF_IDF_USE_IDF = True
TF_IDF_SMOOTH_IDF = True
TF_IDF_SUBLINEAR_TF = True

# Evaluation parameters
EVAL_METRICS = ['precision', 'recall', 'f1', 'map', 'ndcg']
