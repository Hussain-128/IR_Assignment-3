"""
Evaluation Module
Implements metrics for evaluating retrieval system performance
"""
import json
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import config


class Evaluator:
    """Evaluation metrics for information retrieval systems"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics_history = []
    
    @staticmethod
    def precision(retrieved: Set[str], relevant: Set[str]) -> float:
        """
        Calculate precision: fraction of retrieved documents that are relevant
        
        Precision = |retrieved ∩ relevant| / |retrieved|
        
        Args:
            retrieved: Set of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            Precision score [0, 1]
        """
        if not retrieved:
            return 0.0
        
        relevant_retrieved = retrieved.intersection(relevant)
        return len(relevant_retrieved) / len(retrieved)
    
    @staticmethod
    def recall(retrieved: Set[str], relevant: Set[str]) -> float:
        """
        Calculate recall: fraction of relevant documents that are retrieved
        
        Recall = |retrieved ∩ relevant| / |relevant|
        
        Args:
            retrieved: Set of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            Recall score [0, 1]
        """
        if not relevant:
            return 0.0
        
        relevant_retrieved = retrieved.intersection(relevant)
        return len(relevant_retrieved) / len(relevant)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score: harmonic mean of precision and recall
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score [0, 1]
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def average_precision(retrieved_ranked: List[str], relevant: Set[str]) -> float:
        """
        Calculate Average Precision (AP)
        
        AP = (Σ P(k) * rel(k)) / |relevant|
        where P(k) is precision at position k, rel(k) is 1 if doc at k is relevant
        
        Args:
            retrieved_ranked: List of retrieved doc IDs in ranked order
            relevant: Set of relevant document IDs
            
        Returns:
            Average Precision score [0, 1]
        """
        if not relevant or not retrieved_ranked:
            return 0.0
        
        num_relevant = 0
        sum_precision = 0.0
        
        for k, doc_id in enumerate(retrieved_ranked, 1):
            if doc_id in relevant:
                num_relevant += 1
                precision_at_k = num_relevant / k
                sum_precision += precision_at_k
        
        return sum_precision / len(relevant)
    
    @staticmethod
    def mean_average_precision(query_results: Dict[str, List[str]], 
                               ground_truth: Dict[str, Set[str]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries
        
        MAP = (Σ AP(q)) / |queries|
        
        Args:
            query_results: Dict mapping query_id to ranked list of retrieved docs
            ground_truth: Dict mapping query_id to set of relevant docs
            
        Returns:
            MAP score [0, 1]
        """
        if not query_results:
            return 0.0
        
        total_ap = 0.0
        num_queries = 0
        
        for query_id, retrieved in query_results.items():
            if query_id in ground_truth:
                relevant = ground_truth[query_id]
                ap = Evaluator.average_precision(retrieved, relevant)
                total_ap += ap
                num_queries += 1
        
        return total_ap / num_queries if num_queries > 0 else 0.0
    
    @staticmethod
    def dcg(retrieved_ranked: List[str], relevance_scores: Dict[str, float], k: int = None) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG)
        
        DCG@k = Σ (rel_i / log2(i+1)) for i=1 to k
        
        Args:
            retrieved_ranked: List of retrieved doc IDs in ranked order
            relevance_scores: Dict mapping doc_id to relevance score
            k: Cutoff position (None for all)
            
        Returns:
            DCG score
        """
        if k is None:
            k = len(retrieved_ranked)
        
        dcg_score = 0.0
        
        for i, doc_id in enumerate(retrieved_ranked[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg_score += rel / np.log2(i + 1)
        
        return dcg_score
    
    @staticmethod
    def ndcg(retrieved_ranked: List[str], relevance_scores: Dict[str, float], k: int = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)
        
        NDCG@k = DCG@k / IDCG@k
        where IDCG is the ideal (maximum possible) DCG
        
        Args:
            retrieved_ranked: List of retrieved doc IDs in ranked order
            relevance_scores: Dict mapping doc_id to relevance score
            k: Cutoff position (None for all)
            
        Returns:
            NDCG score [0, 1]
        """
        if k is None:
            k = len(retrieved_ranked)
        
        # Calculate actual DCG
        actual_dcg = Evaluator.dcg(retrieved_ranked, relevance_scores, k)
        
        # Calculate ideal DCG (sort by relevance scores)
        ideal_ranking = sorted(relevance_scores.keys(), 
                             key=lambda x: relevance_scores[x], 
                             reverse=True)
        ideal_dcg = Evaluator.dcg(ideal_ranking, relevance_scores, k)
        
        # Normalize
        if ideal_dcg == 0:
            return 0.0
        
        return actual_dcg / ideal_dcg
    
    @staticmethod
    def reciprocal_rank(retrieved_ranked: List[str], relevant: Set[str]) -> float:
        """
        Calculate Reciprocal Rank (RR)
        
        RR = 1 / rank of first relevant document
        
        Args:
            retrieved_ranked: List of retrieved doc IDs in ranked order
            relevant: Set of relevant document IDs
            
        Returns:
            Reciprocal Rank score
        """
        for rank, doc_id in enumerate(retrieved_ranked, 1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(query_results: Dict[str, List[str]], 
                            ground_truth: Dict[str, Set[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) across multiple queries
        
        MRR = (Σ RR(q)) / |queries|
        
        Args:
            query_results: Dict mapping query_id to ranked list of retrieved docs
            ground_truth: Dict mapping query_id to set of relevant docs
            
        Returns:
            MRR score
        """
        if not query_results:
            return 0.0
        
        total_rr = 0.0
        num_queries = 0
        
        for query_id, retrieved in query_results.items():
            if query_id in ground_truth:
                relevant = ground_truth[query_id]
                rr = Evaluator.reciprocal_rank(retrieved, relevant)
                total_rr += rr
                num_queries += 1
        
        return total_rr / num_queries if num_queries > 0 else 0.0
    
    def evaluate_query(self, retrieved: List[str], relevant: Set[str], 
                      relevance_scores: Dict[str, float] = None) -> Dict[str, float]:
        """
        Evaluate a single query with multiple metrics
        
        Args:
            retrieved: List of retrieved doc IDs (ranked)
            relevant: Set of relevant document IDs
            relevance_scores: Optional dict of relevance scores for NDCG
            
        Returns:
            Dict of metric names to scores
        """
        retrieved_set = set(retrieved)
        
        # Calculate basic metrics
        prec = self.precision(retrieved_set, relevant)
        rec = self.recall(retrieved_set, relevant)
        f1 = self.f1_score(prec, rec)
        ap = self.average_precision(retrieved, relevant)
        rr = self.reciprocal_rank(retrieved, relevant)
        
        metrics = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'average_precision': ap,
            'reciprocal_rank': rr
        }
        
        # Calculate NDCG if relevance scores provided
        if relevance_scores:
            ndcg = self.ndcg(retrieved, relevance_scores)
            metrics['ndcg'] = ndcg
        
        return metrics
    
    def evaluate_system(self, query_results: Dict[str, List[str]], 
                       ground_truth: Dict[str, Set[str]],
                       relevance_scores: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
        """
        Evaluate entire system across multiple queries
        
        Args:
            query_results: Dict mapping query_id to ranked list of retrieved docs
            ground_truth: Dict mapping query_id to set of relevant docs
            relevance_scores: Optional dict of query_id -> (doc_id -> score)
            
        Returns:
            Dict of average metric scores
        """
        query_metrics = []
        
        for query_id, retrieved in query_results.items():
            if query_id not in ground_truth:
                continue
            
            relevant = ground_truth[query_id]
            rel_scores = relevance_scores.get(query_id) if relevance_scores else None
            
            metrics = self.evaluate_query(retrieved, relevant, rel_scores)
            metrics['query_id'] = query_id
            query_metrics.append(metrics)
        
        # Calculate averages
        if not query_metrics:
            return {}
        
        avg_metrics = {
            'num_queries': len(query_metrics),
            'avg_precision': np.mean([m['precision'] for m in query_metrics]),
            'avg_recall': np.mean([m['recall'] for m in query_metrics]),
            'avg_f1': np.mean([m['f1'] for m in query_metrics]),
            'map': np.mean([m['average_precision'] for m in query_metrics]),
            'mrr': np.mean([m['reciprocal_rank'] for m in query_metrics])
        }
        
        # Add NDCG if available
        if relevance_scores and 'ndcg' in query_metrics[0]:
            avg_metrics['avg_ndcg'] = np.mean([m['ndcg'] for m in query_metrics])
        
        # Store detailed results
        self.metrics_history.append({
            'summary': avg_metrics,
            'per_query': query_metrics
        })
        
        return avg_metrics
    
    def print_evaluation(self, metrics: Dict[str, float], detailed: bool = False):
        """
        Print evaluation metrics in a readable format
        
        Args:
            metrics: Dict of metric scores
            detailed: Whether to print detailed per-query results
        """
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\nNumber of queries evaluated: {metrics.get('num_queries', 0)}")
        print("\nAverage Metrics:")
        print("-" * 40)
        print(f"Precision: {metrics.get('avg_precision', 0):.4f}")
        print(f"Recall:    {metrics.get('avg_recall', 0):.4f}")
        print(f"F1 Score:  {metrics.get('avg_f1', 0):.4f}")
        print(f"MAP:       {metrics.get('map', 0):.4f}")
        print(f"MRR:       {metrics.get('mrr', 0):.4f}")
        
        if 'avg_ndcg' in metrics:
            print(f"NDCG:      {metrics.get('avg_ndcg', 0):.4f}")
        
        if detailed and self.metrics_history:
            print("\n" + "=" * 70)
            print("PER-QUERY RESULTS")
            print("=" * 70)
            
            per_query = self.metrics_history[-1]['per_query']
            for m in per_query:
                print(f"\nQuery: {m['query_id']}")
                print(f"  Precision: {m['precision']:.4f}")
                print(f"  Recall:    {m['recall']:.4f}")
                print(f"  F1:        {m['f1']:.4f}")
                print(f"  AP:        {m['average_precision']:.4f}")
    
    def save_results(self, filepath: str):
        """
        Save evaluation results to JSON file
        
        Args:
            filepath: Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")


if __name__ == "__main__":
    # Test evaluation metrics
    evaluator = Evaluator()
    
    # Example: Single query evaluation
    print("Testing single query evaluation:")
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = {'doc1', 'doc3', 'doc6'}
    
    metrics = evaluator.evaluate_query(retrieved, relevant)
    print(json.dumps(metrics, indent=2))
    
    # Example: Multiple queries evaluation
    print("\n" + "=" * 70)
    print("Testing multiple queries evaluation:")
    
    query_results = {
        'q1': ['doc1', 'doc2', 'doc3'],
        'q2': ['doc4', 'doc5', 'doc6'],
        'q3': ['doc1', 'doc7', 'doc8']
    }
    
    ground_truth = {
        'q1': {'doc1', 'doc3'},
        'q2': {'doc4', 'doc6'},
        'q3': {'doc1', 'doc2'}
    }
    
    system_metrics = evaluator.evaluate_system(query_results, ground_truth)
    evaluator.print_evaluation(system_metrics)
