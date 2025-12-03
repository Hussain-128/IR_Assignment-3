"""
Document Indexing Module
Builds and manages the inverted index and TF-IDF vectors
"""
import os
import pickle
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import config
from preprocessing import Preprocessor


class DocumentIndex:
    """Manages document indexing and storage"""
    
    def __init__(self):
        """Initialize the document index"""
        self.preprocessor = Preprocessor()
        self.documents = {}  # doc_id -> document text
        self.doc_ids = []    # List of document IDs
        self.doc_metadata = {}  # doc_id -> metadata (filename, path, etc.)
        
        # Inverted index: term -> list of (doc_id, frequency)
        self.inverted_index = defaultdict(list)
        
        # TF-IDF components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.vocabulary = None
        
        # Statistics
        self.total_docs = 0
        self.avg_doc_length = 0
        self.doc_lengths = {}  # doc_id -> document length
    
    def load_documents_from_directory(self, directory: str) -> int:
        """
        Load all text documents from a directory
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            Number of documents loaded
        """
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
            print(f"Please place your .txt documents in {directory}")
            return 0
        
        doc_count = 0
        
        print(f"Loading documents from {directory}...")
        
        # Recursively find all .txt files
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.txt'):
                    filepath = os.path.join(root, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Use filename (without extension) as doc_id
                        doc_id = os.path.splitext(filename)[0]
                        
                        # Store document
                        self.documents[doc_id] = content
                        self.doc_metadata[doc_id] = {
                            'filename': filename,
                            'path': filepath,
                            'size': len(content)
                        }
                        
                        doc_count += 1
                    
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        self.doc_ids = list(self.documents.keys())
        self.total_docs = len(self.doc_ids)
        
        print(f"Loaded {doc_count} documents")
        return doc_count
    
    def build_inverted_index(self):
        """Build inverted index from documents"""
        print("\nBuilding inverted index...")
        
        self.inverted_index = defaultdict(lambda: defaultdict(int))
        total_length = 0
        
        for doc_id in tqdm(self.doc_ids, desc="Processing documents"):
            # Preprocess document
            tokens = self.preprocessor.preprocess_document(self.documents[doc_id])
            
            # Count term frequencies
            term_freq = Counter(tokens)
            
            # Store document length
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # Update inverted index
            for term, freq in term_freq.items():
                self.inverted_index[term][doc_id] = freq
        
        # Calculate average document length
        if self.total_docs > 0:
            self.avg_doc_length = total_length / self.total_docs
        
        print(f"Inverted index built with {len(self.inverted_index)} unique terms")
        print(f"Average document length: {self.avg_doc_length:.2f} tokens")
    
    def build_tfidf_index(self):
        """Build TF-IDF vectors for all documents"""
        print("\nBuilding TF-IDF index...")
        
        # Preprocess all documents
        processed_docs = []
        
        for doc_id in tqdm(self.doc_ids, desc="Preprocessing documents"):
            tokens = self.preprocessor.preprocess_document(self.documents[doc_id])
            # Join tokens back into string for TfidfVectorizer
            processed_docs.append(' '.join(tokens))
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            norm=config.TF_IDF_NORM,
            use_idf=config.TF_IDF_USE_IDF,
            smooth_idf=config.TF_IDF_SMOOTH_IDF,
            sublinear_tf=config.TF_IDF_SUBLINEAR_TF,
            lowercase=False  # Already lowercased
        )
        
        # Fit and transform documents
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_docs)
        self.vocabulary = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"TF-IDF index built:")
        print(f"  - Vocabulary size: {len(self.vocabulary)}")
        print(f"  - Matrix shape: {self.tfidf_matrix.shape}")
        print(f"  - Matrix density: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4%}")
    
    def build_index(self):
        """Build complete index (inverted index + TF-IDF)"""
        if self.total_docs == 0:
            print("No documents to index!")
            return
        
        self.build_inverted_index()
        
        if config.USE_TF_IDF:
            self.build_tfidf_index()
        
        print("\n✓ Index building complete!")
    
    def save_index(self, index_dir: str = None):
        """
        Save index to disk
        
        Args:
            index_dir: Directory to save index files
        """
        if index_dir is None:
            index_dir = config.INDEX_DIR
        
        os.makedirs(index_dir, exist_ok=True)
        
        print(f"\nSaving index to {index_dir}...")
        
        # Save inverted index
        with open(os.path.join(index_dir, 'inverted_index.pkl'), 'wb') as f:
            pickle.dump(dict(self.inverted_index), f)
        
        # Save document metadata
        with open(os.path.join(index_dir, 'doc_metadata.json'), 'w') as f:
            json.dump({
                'doc_ids': self.doc_ids,
                'metadata': self.doc_metadata,
                'total_docs': self.total_docs,
                'avg_doc_length': self.avg_doc_length,
                'doc_lengths': self.doc_lengths
            }, f, indent=2)
        
        # Save TF-IDF components
        if self.tfidf_vectorizer is not None:
            with open(os.path.join(index_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            with open(os.path.join(index_dir, 'tfidf_matrix.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_matrix, f)
        
        print("✓ Index saved successfully!")
    
    def load_index(self, index_dir: str = None):
        """
        Load index from disk
        
        Args:
            index_dir: Directory containing index files
        """
        if index_dir is None:
            index_dir = config.INDEX_DIR
        
        print(f"Loading index from {index_dir}...")
        
        try:
            # Load inverted index
            with open(os.path.join(index_dir, 'inverted_index.pkl'), 'rb') as f:
                self.inverted_index = defaultdict(lambda: defaultdict(int), pickle.load(f))
            
            # Load document metadata
            with open(os.path.join(index_dir, 'doc_metadata.json'), 'r') as f:
                metadata = json.load(f)
                self.doc_ids = metadata['doc_ids']
                self.doc_metadata = metadata['metadata']
                self.total_docs = metadata['total_docs']
                self.avg_doc_length = metadata['avg_doc_length']
                self.doc_lengths = metadata['doc_lengths']
            
            # Load TF-IDF components
            tfidf_vectorizer_path = os.path.join(index_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_vectorizer_path):
                with open(tfidf_vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                
                with open(os.path.join(index_dir, 'tfidf_matrix.pkl'), 'rb') as f:
                    self.tfidf_matrix = pickle.load(f)
                
                self.vocabulary = self.tfidf_vectorizer.get_feature_names_out()
            
            print("✓ Index loaded successfully!")
            print(f"  - Total documents: {self.total_docs}")
            print(f"  - Vocabulary size: {len(self.inverted_index)}")
            
        except FileNotFoundError:
            print("Index files not found. Please build the index first.")
            raise
    
    def get_document_frequency(self, term: str) -> int:
        """Get number of documents containing a term"""
        return len(self.inverted_index.get(term, {}))
    
    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Get frequency of term in a specific document"""
        return self.inverted_index.get(term, {}).get(doc_id, 0)
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        return {
            'total_documents': self.total_docs,
            'vocabulary_size': len(self.inverted_index),
            'avg_doc_length': self.avg_doc_length,
            'total_tokens': sum(self.doc_lengths.values()),
            'tfidf_enabled': self.tfidf_matrix is not None
        }


if __name__ == "__main__":
    # Test the indexer
    indexer = DocumentIndex()
    
    # Load documents
    doc_dir = os.path.join(config.DATA_DIR, 'documents')
    indexer.load_documents_from_directory(doc_dir)
    
    if indexer.total_docs > 0:
        # Build index
        indexer.build_index()
        
        # Save index
        indexer.save_index()
        
        # Print statistics
        print("\nIndex Statistics:")
        print(json.dumps(indexer.get_index_stats(), indent=2))
