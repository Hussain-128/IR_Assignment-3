"""
Main Application Interface
Command-line interface for the Information Retrieval System
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
from typing import Dict, Set, List, Tuple
import config
from indexer import DocumentIndex
from retrieval import RetrievalEngine
from evaluation import Evaluator


class IRSystem:
    """Main Information Retrieval System"""
    
    def __init__(self):
        """Initialize IR system"""
        self.index = DocumentIndex()
        self.retrieval_engine = None
        self.evaluator = Evaluator()
    
    def build_index(self, doc_directory: str = None):
        """
        Build index from documents
        
        Args:
            doc_directory: Directory containing documents
        """
        if doc_directory is None:
            doc_directory = os.path.join(config.DATA_DIR, 'documents')
        
        print("=" * 70)
        print("BUILDING INDEX")
        print("=" * 70)
        
        # Load documents
        num_docs = self.index.load_documents_from_directory(doc_directory)
        
        if num_docs == 0:
            print("\nNo documents found!")
            print(f"Please place your .txt documents in: {doc_directory}")
            return False
        
        # Build index
        self.index.build_index()
        
        # Save index
        self.index.save_index()
        
        # Print statistics
        stats = self.index.get_index_stats()
        print("\nIndex Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return True
    
    def load_index(self):
        """Load existing index"""
        try:
            self.index.load_index()
            self.retrieval_engine = RetrievalEngine(self.index)
            return True
        except FileNotFoundError:
            print("Error: Index not found!")
            print("Please build the index first using: python main.py --build-index")
            return False
    
    def search(self, query: str, method: str = 'tfidf', top_k: int = None, 
               show_snippets: bool = True, interactive_eval: bool = False):
        """
        Search for documents
        
        Args:
            query: Search query
            method: Retrieval method
            top_k: Number of results
            show_snippets: Show document snippets
            interactive_eval: Enable interactive evaluation mode
        """
        if self.retrieval_engine is None:
            if not self.load_index():
                return
        
        print("\n" + "=" * 70)
        print(f"SEARCHING: {query}")
        print(f"Method: {method.upper()}")
        print("=" * 70)
        
        # Perform search
        results = self.retrieval_engine.search(query, method=method, top_k=top_k)
        
        # Format and print results
        formatted = self.retrieval_engine.format_results(query, results, show_snippets)
        print("\n" + formatted)
        
        # Interactive evaluation mode
        if interactive_eval:
            self._interactive_evaluation(query, results)
    
    def _interactive_evaluation(self, query: str, results: List[Tuple[str, float]]):
        """
        Interactive evaluation mode - asks user for relevant documents
        
        Args:
            query: The search query
            results: Retrieved results [(doc_id, score), ...]
        """
        print("\n" + "=" * 70)
        print("OPTIONS:")
        print("  1. Evaluate this query (Calculate Precision, Recall, etc.)")
        print("  2. Exit")
        print("=" * 70)
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '2':
            print("\nExiting...")
            import sys
            sys.exit(0)
        
        if choice != '1':
            return
        
        # Get relevant document IDs from user
        print("\n" + "-" * 70)
        print("Please enter the relevant document IDs for this query")
        print("(comma-separated, e.g., article_0001_1-1-2015, article_0002_1-2-2015)")
        print("Hint: Look at the document IDs shown in the search results above")
        print("-" * 70)
        
        relevant_input = input("\nRelevant document IDs: ").strip()
        
        if not relevant_input:
            print("\n✗ No relevant documents provided. Evaluation cancelled.")
            return
        
        # Parse the relevant document IDs  
        relevant_docs = [doc.strip() for doc in relevant_input.split(',')]
        relevant_set = set(relevant_docs)
        
        # Calculate metrics
        retrieved_set = set([doc_id for doc_id, _ in results])
        
        true_positives = retrieved_set.intersection(relevant_set)
        false_positives = retrieved_set - relevant_set
        false_negatives = relevant_set - retrieved_set
        
        precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0
        recall = len(true_positives) / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display results
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\nQuery: '{query}'")
        print(f"Retrieved documents: {len(retrieved_set)}")
        print(f"Relevant documents (provided): {len(relevant_set)}")
        
        print(f"\nTrue Positives (Relevant & Retrieved): {len(true_positives)}")
        print(f"False Positives (Retrieved but Not Relevant): {len(false_positives)}")
        print(f"False Negatives (Relevant but Not Retrieved): {len(false_negatives)}")
        
        print("\n" + "-" * 70)
        print(f"Precision: {precision:.4f} ({len(true_positives)}/{len(retrieved_set)})")
        print(f"  → {precision*100:.2f}% of retrieved documents are relevant")
        
        print(f"\nRecall:    {recall:.4f} ({len(true_positives)}/{len(relevant_set)})")
        print(f"  → {recall*100:.2f}% of relevant documents were retrieved")
        
        print(f"\nF1 Score:  {f1:.4f}")
        print(f"  → Harmonic mean of Precision and Recall")
        print("=" * 70)
        
        if len(true_positives) > 0:
            print(f"\n✓ Correctly Retrieved Documents:")
            for doc in true_positives:
                print(f"  • {doc}")
        
        if len(false_positives) > 0:
            print(f"\n✗ Incorrectly Retrieved Documents:")
            for doc in list(false_positives)[:5]:
                print(f"  • {doc}")
            if len(false_positives) > 5:
                print(f"  ... and {len(false_positives) - 5} more")
        
        if len(false_negatives) > 0:
            print(f"\n⚠ Missed Relevant Documents:")
            for doc in list(false_negatives)[:5]:
                print(f"  • {doc}")
            if len(false_negatives) > 5:
                print(f"  ... and {len(false_negatives) - 5} more")
    
    def interactive_search(self, method: str = 'tfidf'):
        """Interactive search mode"""
        if self.retrieval_engine is None:
            if not self.load_index():
                return
        
        print("\n" + "=" * 70)
        print("INTERACTIVE SEARCH MODE")
        print("=" * 70)
        print(f"Retrieval method: {method.upper()}")
        print("Enter your queries (type 'exit' or 'quit' to stop)")
        print("=" * 70)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Exiting...")
                    break
                
                if not query:
                    continue
                
                # Perform search
                results = self.retrieval_engine.search(query, method=method)
                formatted = self.retrieval_engine.format_results(query, results, 
                                                                show_snippets=True)
                print("\n" + formatted)
                
                # Ask for evaluation
                self._interactive_evaluation(query, results)
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def evaluate(self, queries_file: str):
        """
        Evaluate system using ground truth queries
        
        Args:
            queries_file: JSON file with queries and relevance judgments
        """
        if self.retrieval_engine is None:
            if not self.load_index():
                return
        
        print("\n" + "=" * 70)
        print("EVALUATING SYSTEM")
        print("=" * 70)
        
        # Load queries and ground truth
        try:
            with open(queries_file, 'r') as f:
                eval_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Queries file not found: {queries_file}")
            print("\nExpected format:")
            print(self.get_example_queries_format())
            return
        
        queries = eval_data.get('queries', {})
        ground_truth = eval_data.get('ground_truth', {})
        relevance_scores = eval_data.get('relevance_scores', None)
        
        if not queries or not ground_truth:
            print("Error: Invalid queries file format!")
            print("\nExpected format:")
            print(self.get_example_queries_format())
            return
        
        print(f"\nEvaluating {len(queries)} queries...")
        
        # Run queries and collect results
        query_results = {}
        
        for query_id, query_text in queries.items():
            results = self.retrieval_engine.search(query_text, method='tfidf')
            query_results[query_id] = [doc_id for doc_id, _ in results]
        
        # Convert ground truth to sets
        ground_truth_sets = {
            qid: set(docs) if isinstance(docs, list) else docs 
            for qid, docs in ground_truth.items()
        }
        
        # Evaluate
        metrics = self.evaluator.evaluate_system(
            query_results, 
            ground_truth_sets, 
            relevance_scores
        )
        
        # Print results
        self.evaluator.print_evaluation(metrics, detailed=True)
        
        # Save results
        results_file = os.path.join(config.RESULTS_DIR, 'evaluation_results.json')
        self.evaluator.save_results(results_file)
    
    @staticmethod
    def get_example_queries_format() -> str:
        """Get example format for queries file"""
        example = {
            "queries": {
                "q1": "information retrieval",
                "q2": "search engines"
            },
            "ground_truth": {
                "q1": ["doc1", "doc3", "doc5"],
                "q2": ["doc2", "doc4"]
            },
            "relevance_scores": {
                "q1": {"doc1": 3, "doc3": 2, "doc5": 1},
                "q2": {"doc2": 2, "doc4": 1}
            }
        }
        return json.dumps(example, indent=2)
    
    def create_sample_documents(self):
        """Create sample documents for testing"""
        doc_dir = os.path.join(config.DATA_DIR, 'documents')
        os.makedirs(doc_dir, exist_ok=True)
        
        sample_docs = {
            "doc1_ir_basics.txt": """
            Information Retrieval Systems
            
            Information retrieval (IR) is the process of obtaining information system resources 
            that are relevant to an information need from a collection of those resources. 
            Searches can be based on full-text or other content-based indexing.
            
            The most common type of information retrieval system is a search engine, which 
            allows users to find documents or web pages containing specific keywords or phrases.
            
            Key concepts in IR include:
            - Document representation
            - Query processing
            - Relevance ranking
            - Evaluation metrics
            """,
            
            "doc2_search_engines.txt": """
            Modern Search Engines
            
            Search engines are complex information retrieval systems that index billions of 
            web pages and provide fast, relevant results to user queries.
            
            Popular search engines include Google, Bing, and DuckDuckGo. These systems use 
            sophisticated algorithms to rank documents based on relevance, popularity, and 
            many other factors.
            
            The basic components of a search engine are:
            1. Web crawler (spider)
            2. Indexer
            3. Query processor
            4. Ranking algorithm
            """,
            
            "doc3_vector_space_model.txt": """
            Vector Space Model in Information Retrieval
            
            The vector space model (VSM) is an algebraic model for representing text documents 
            as vectors of identifiers. It is used in information filtering, information retrieval, 
            indexing and relevance rankings.
            
            In the VSM, documents and queries are represented as vectors in a high-dimensional 
            space where each dimension corresponds to a term from the vocabulary.
            
            The similarity between documents and queries is computed using measures like 
            cosine similarity. TF-IDF (Term Frequency-Inverse Document Frequency) is commonly 
            used for term weighting in the vector space model.
            """,
            
            "doc4_machine_learning.txt": """
            Machine Learning Applications
            
            Machine learning is a branch of artificial intelligence that focuses on building 
            systems that can learn from data. Modern applications include image recognition, 
            natural language processing, and recommendation systems.
            
            Deep learning, a subset of machine learning, uses neural networks with many layers 
            to learn complex patterns in large amounts of data.
            
            Common machine learning algorithms include:
            - Linear regression
            - Decision trees
            - Neural networks
            - Support vector machines
            """,
            
            "doc5_tfidf.txt": """
            TF-IDF: Term Frequency-Inverse Document Frequency
            
            TF-IDF is a numerical statistic used in information retrieval to reflect how 
            important a word is to a document in a collection of documents.
            
            The TF-IDF value increases proportionally to the number of times a word appears 
            in the document (term frequency) and is offset by the number of documents in the 
            corpus that contain the word (inverse document frequency).
            
            This helps to adjust for the fact that some words appear more frequently in general, 
            and thus are less informative for distinguishing documents.
            
            TF-IDF is widely used in search engines and document classification systems.
            """
        }
        
        print("Creating sample documents...")
        for filename, content in sample_docs.items():
            filepath = os.path.join(doc_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"  Created: {filename}")
        
        print(f"\n✓ Created {len(sample_docs)} sample documents in {doc_dir}")
    
    def create_sample_queries(self):
        """Create sample queries file for evaluation"""
        queries_file = os.path.join(config.DATA_DIR, 'sample_queries.json')
        
        sample_data = {
            "queries": {
                "q1": "information retrieval systems",
                "q2": "search engine algorithms",
                "q3": "vector space model",
                "q4": "tf-idf weighting"
            },
            "ground_truth": {
                "q1": ["doc1_ir_basics", "doc2_search_engines"],
                "q2": ["doc2_search_engines"],
                "q3": ["doc3_vector_space_model", "doc5_tfidf"],
                "q4": ["doc5_tfidf", "doc3_vector_space_model"]
            },
            "relevance_scores": {
                "q1": {"doc1_ir_basics": 3, "doc2_search_engines": 2},
                "q2": {"doc2_search_engines": 3},
                "q3": {"doc3_vector_space_model": 3, "doc5_tfidf": 1},
                "q4": {"doc5_tfidf": 3, "doc3_vector_space_model": 2}
            }
        }
        
        with open(queries_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✓ Created sample queries file: {queries_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Information Retrieval System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample documents and queries
  python main.py --create-samples
  
  # Build index from documents
  python main.py --build-index
  
  # Search with a query
  python main.py --query "information retrieval"
  
  # Interactive search mode
  python main.py --interactive
  
  # Evaluate with ground truth
  python main.py --evaluate --queries-file data/sample_queries.json
  
  # Search using BM25
  python main.py --query "search engines" --method bm25
        """
    )
    
    # Actions
    parser.add_argument('--build-index', action='store_true',
                       help='Build index from documents')
    parser.add_argument('--query', type=str,
                       help='Search query')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive search mode')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate system with ground truth')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample documents and queries')
    
    # Options
    parser.add_argument('--method', type=str, default='tfidf',
                       choices=['tfidf', 'bm25', 'boolean'],
                       help='Retrieval method (default: tfidf)')
    parser.add_argument('--top-k', type=int, default=config.TOP_K_RESULTS,
                       help=f'Number of results (default: {config.TOP_K_RESULTS})')
    parser.add_argument('--doc-dir', type=str,
                       help='Document directory')
    parser.add_argument('--queries-file', type=str,
                       help='Queries file for evaluation')
    parser.add_argument('--no-snippets', action='store_true',
                       help='Disable document snippets in results')
    
    args = parser.parse_args()
    
    # Create IR system
    ir_system = IRSystem()
    
    # Execute requested action
    if args.create_samples:
        ir_system.create_sample_documents()
        ir_system.create_sample_queries()
    
    elif args.build_index:
        ir_system.build_index(args.doc_dir)
    
    elif args.query:
        ir_system.search(
            args.query, 
            method=args.method, 
            top_k=args.top_k,
            show_snippets=not args.no_snippets,
            interactive_eval=True
        )
    
    elif args.interactive:
        ir_system.interactive_search(method=args.method)
    
    elif args.evaluate:
        if not args.queries_file:
            print("Error: --queries-file required for evaluation")
            print("\nExample: python main.py --evaluate --queries-file data/sample_queries.json")
            sys.exit(1)
        ir_system.evaluate(args.queries_file)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
