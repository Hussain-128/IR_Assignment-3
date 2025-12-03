"""
Document Preprocessing Module
Handles text cleaning, tokenization, normalization, and stemming
"""
import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import config


class Preprocessor:
    """Text preprocessor for documents and queries"""
    
    def __init__(self):
        """Initialize preprocessor with NLTK components"""
        self.stemmer = PorterStemmer() if config.USE_STEMMING else None
        self.lemmatizer = WordNetLemmatizer() if config.USE_LEMMATIZATION else None
        
        # Load stopwords
        if config.REMOVE_STOPWORDS:
            try:
                self.stopwords = set(stopwords.words('english'))
            except LookupError:
                print("Warning: NLTK stopwords not found. Run setup_nltk.py first.")
                self.stopwords = set()
        else:
            self.stopwords = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback to simple split if NLTK tokenizer not available
            tokens = text.split()
        
        return tokens
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by length and stopwords
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        filtered = []
        
        for token in tokens:
            # Check length
            if len(token) < config.MIN_WORD_LENGTH or len(token) > config.MAX_WORD_LENGTH:
                continue
            
            # Check stopwords
            if config.REMOVE_STOPWORDS and token in self.stopwords:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens using stemming or lemmatization
        
        Args:
            tokens: List of tokens
            
        Returns:
            Normalized list of tokens
        """
        if config.USE_STEMMING and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif config.USE_LEMMATIZATION and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw text string
            
        Returns:
            List of preprocessed tokens
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Filter
        tokens = self.filter_tokens(tokens)
        
        # Normalize
        tokens = self.normalize_tokens(tokens)
        
        return tokens
    
    def preprocess_document(self, doc_text: str) -> List[str]:
        """
        Preprocess a document
        
        Args:
            doc_text: Raw document text
            
        Returns:
            List of preprocessed tokens
        """
        return self.preprocess(doc_text)
    
    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess a query (same as document preprocessing)
        
        Args:
            query: Raw query string
            
        Returns:
            List of preprocessed tokens
        """
        return self.preprocess(query)
    
    def get_vocabulary(self, token_lists: List[List[str]]) -> Set[str]:
        """
        Extract unique vocabulary from multiple token lists
        
        Args:
            token_lists: List of token lists
            
        Returns:
            Set of unique tokens
        """
        vocabulary = set()
        for tokens in token_lists:
            vocabulary.update(tokens)
        return vocabulary


def test_preprocessor():
    """Test the preprocessor with sample text"""
    preprocessor = Preprocessor()
    
    sample_texts = [
        "This is a SIMPLE test! With some HTML <b>tags</b> and https://example.com URLs.",
        "The quick brown fox jumps over the lazy dog.",
        "Information retrieval systems are used to find relevant documents."
    ]
    
    print("Testing Preprocessor:")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}:")
        print(f"Original: {text}")
        tokens = preprocessor.preprocess(text)
        print(f"Processed: {tokens}")
        print(f"Token count: {len(tokens)}")


if __name__ == "__main__":
    test_preprocessor()
