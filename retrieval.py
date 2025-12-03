"""
Retrieval Module
Handles query processing and document retrieval
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import config
from preprocessing import Preprocessor
from indexer import DocumentIndex


class RetrievalEngine:
    """Query processing and document retrieval engine"""
    
    def __init__(self, index: DocumentIndex):
        """
        Initialize retrieval engine
        
        Args:
            index: Document index to search
        """
        self.index = index
        self.preprocessor = Preprocessor()
    
    def search_tfidf(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Search using TF-IDF cosine similarity
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        if self.index.tfidf_vectorizer is None:
            raise ValueError("TF-IDF index not built. Set USE_TF_IDF=True in config.")
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_query(query)
        query_text = ' '.join(query_tokens)
        
        # Transform query to TF-IDF vector
        query_vector = self.index.tfidf_vectorizer.transform([query_text])
        
        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_vector, self.index.tfidf_matrix)[0]
        
        # Get top-k documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self.index.doc_ids[idx]
            score = similarities[idx]
            if score > 0:  # Only include documents with non-zero similarity
                results.append((doc_id, float(score)))
        
        return results
    
    def search_bm25(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Search using BM25 ranking function
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_query(query)
        
        # BM25 parameters
        k1 = config.BM25_K1
        b = config.BM25_B
        N = self.index.total_docs
        avgdl = self.index.avg_doc_length
        
        # Calculate BM25 scores for each document
        scores = {}
        
        for doc_id in self.index.doc_ids:
            score = 0.0
            doc_length = self.index.doc_lengths.get(doc_id, 0)
            
            for term in query_tokens:
                if term not in self.index.inverted_index:
                    continue
                
                # Term frequency in document
                tf = self.index.get_term_frequency(term, doc_id)
                
                if tf == 0:
                    continue
                
                # Document frequency
                df = self.index.get_document_frequency(term)
                
                # IDF component
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
                
                # BM25 score for this term
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
                
                score += idf * (numerator / denominator)
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and get top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results
    
    def search_boolean(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Simple Boolean retrieval (AND operation)
        Returns documents containing all query terms
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_query(query)
        
        if not query_tokens:
            return []
        
        # Find documents containing first term
        first_term = query_tokens[0]
        if first_term not in self.index.inverted_index:
            return []
        
        candidate_docs = set(self.index.inverted_index[first_term].keys())
        
        # Intersect with documents containing other terms (AND operation)
        for term in query_tokens[1:]:
            if term in self.index.inverted_index:
                term_docs = set(self.index.inverted_index[term].keys())
                candidate_docs = candidate_docs.intersection(term_docs)
            else:
                return []  # If any term is not found, no documents match
        
        # Score by number of term occurrences
        scores = {}
        for doc_id in candidate_docs:
            score = sum(self.index.get_term_frequency(term, doc_id) 
                       for term in query_tokens)
            scores[doc_id] = float(score)
        
        # Sort and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results
    
    def search(self, query: str, method: str = 'tfidf', top_k: int = None) -> List[Tuple[str, float]]:
        """
        Search documents using specified method
        
        Args:
            query: Query string
            method: Retrieval method ('tfidf', 'bm25', or 'boolean')
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        if not query or not query.strip():
            return []
        
        if method == 'tfidf':
            return self.search_tfidf(query, top_k)
        elif method == 'bm25':
            return self.search_bm25(query, top_k)
        elif method == 'boolean':
            return self.search_boolean(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def get_document_snippet(self, doc_id: str, query: str, 
                           snippet_length: int = 200) -> str:
        """
        Get a snippet of the document around query terms
        
        Args:
            doc_id: Document ID
            query: Query string
            snippet_length: Maximum snippet length
            
        Returns:
            Document snippet
        """
        if doc_id not in self.index.doc_metadata:
            return ""
        
        # Load document
        doc_path = self.index.doc_metadata[doc_id]['path']
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()
        except:
            return ""
        
        # If document is short, return it all
        if len(doc_text) <= snippet_length:
            return doc_text
        
        # Preprocess query to get terms
        query_tokens = self.preprocessor.preprocess_query(query)
        
        # Find first occurrence of any query term
        doc_lower = doc_text.lower()
        best_pos = 0
        
        for token in query_tokens:
            pos = doc_lower.find(token)
            if pos != -1:
                best_pos = pos
                break
        
        # Extract snippet around that position
        start = max(0, best_pos - snippet_length // 2)
        end = min(len(doc_text), start + snippet_length)
        
        snippet = doc_text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(doc_text):
            snippet = snippet + "..."
        
        return snippet
    
    def format_results(self, query: str, results: List[Tuple[str, float]], 
                      show_snippets: bool = True) -> str:
        """
        Format search results for display
        
        Args:
            query: Original query
            results: List of (doc_id, score) tuples
            show_snippets: Whether to show document snippets
            
        Returns:
            Formatted results string
        """
        if not results:
            return "No results found."
        
        output = []
        output.append(f"Found {len(results)} results for query: '{query}'")
        output.append("=" * 70)
        
        for rank, (doc_id, score) in enumerate(results, 1):
            output.append(f"\n{rank}. Document: {doc_id}")
            output.append(f"   Score: {score:.4f}")
            
            if doc_id in self.index.doc_metadata:
                metadata = self.index.doc_metadata[doc_id]
                output.append(f"   File: {metadata['filename']}")
            
            if show_snippets:
                snippet = self.get_document_snippet(doc_id, query)
                if snippet:
                    output.append(f"   Snippet: {snippet[:200]}")
            
            output.append("-" * 70)
        
        return "\n".join(output)


if __name__ == "__main__":
    # Test the retrieval engine
    import os
    
    # Load index
    index = DocumentIndex()
    try:
        index.load_index()
        
        # Create retrieval engine
        engine = RetrievalEngine(index)
        
        # Test queries
        test_queries = [
            "information retrieval",
            "search engine",
            "document ranking"
        ]
        
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print('='*70)
            
            results = engine.search(query, method='tfidf')
            print(engine.format_results(query, results, show_snippets=False))
    
    except FileNotFoundError:
        print("Index not found. Please build the index first by running indexer.py")
