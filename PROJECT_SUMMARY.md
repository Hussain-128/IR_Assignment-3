# Information Retrieval System - Project Summary

## ✅ Project Completed Successfully

### Dataset
- **Source**: Articles.csv (Business News Dataset)
- **Total Documents**: 2,692 news articles
- **Content**: Pakistani business news (oil prices, stock market, economy, energy, etc.)
- **Date Range**: 2015-2017

### System Implementation

#### 1. **Document Processing**
- Converted CSV to 2,692 individual text documents
- Each document contains: Title, Date, Category, and Article text
- Total vocabulary: 17,546 unique terms
- Average document length: 176.49 tokens

#### 2. **Indexing**
- **Inverted Index**: Built for efficient term lookup
- **TF-IDF Vectors**: 2,692 × 17,546 matrix (0.69% density)
- **Storage**: Index saved to disk for fast loading

#### 3. **Retrieval Methods Implemented**
- ✅ **TF-IDF** (Vector Space Model with cosine similarity)
- ✅ **BM25** (Probabilistic ranking function)
- ✅ **Boolean** (AND-based retrieval)

#### 4. **Preprocessing Pipeline**
- Text cleaning (URLs, special characters removal)
- Tokenization using NLTK
- Stopword removal
- Stemming (Porter Stemmer)
- Configurable parameters

#### 5. **Evaluation Metrics**
- Precision: 0.16
- Recall: 0.40
- F1 Score: 0.22
- MAP (Mean Average Precision): 0.36
- MRR (Mean Reciprocal Rank): 0.40
- NDCG: 0.38

### Example Queries & Results

#### Query 1: "oil prices crude petroleum"
**Top Results:**
1. Article about oil price increase due to Yemen turmoil (Score: 0.2928)
2. Petrol and diesel price hike news (Score: 0.2769)
3. Oil price regulation news (Score: 0.2764)

#### Query 2: "stock market karachi exchange"
**Top Results:**
1. KSE down 1,419 points (Score: 0.3439)
2. US Ambassador visits Stock Exchange (Score: 0.3353)
3. Dollar reaches 17-month high (Score: 0.3210)

#### Query 3: "electricity power load shedding" (BM25)
**Top Results:**
1. K-Electric load shedding news (Score: 26.53)
2. 9000MW electricity to be added (Score: 20.26)
3. PTI resolution against K-Electric (Score: 19.82)

### Key Features

✅ **Local Implementation** - Runs entirely on local machine
✅ **No Cloud Dependencies** - All processing done locally
✅ **Multiple Retrieval Strategies** - TF-IDF, BM25, Boolean
✅ **Efficient Indexing** - Fast search across 2,692+ documents
✅ **Comprehensive Evaluation** - Multiple metrics implemented
✅ **Reproducible** - Clear documentation and setup instructions
✅ **Scalable** - Can handle larger document collections

### Usage Examples

```bash
# Build index from news articles
py -3.11 main.py --build-index

# Search with TF-IDF
py -3.11 main.py --query "oil prices"

# Search with BM25
py -3.11 main.py --query "stock market" --method bm25

# Interactive mode
py -3.11 main.py --interactive

# Evaluate system
py -3.11 main.py --evaluate --queries-file data/news_queries.json
```

### System Architecture

```
CSV File (2,692 articles)
    ↓
Document Conversion (convert_csv.py)
    ↓
Preprocessing (cleaning, tokenization, stemming)
    ↓
Indexing (inverted index + TF-IDF matrix)
    ↓
Query Processing
    ↓
Retrieval & Ranking (TF-IDF/BM25/Boolean)
    ↓
Results with Relevance Scores
    ↓
Evaluation (Precision, Recall, MAP, NDCG)
```

### Files Created

1. **config.py** - Configuration parameters
2. **preprocessing.py** - Text preprocessing module
3. **indexer.py** - Document indexing system
4. **retrieval.py** - Query processing and retrieval
5. **evaluation.py** - Evaluation metrics
6. **main.py** - Main CLI interface
7. **convert_csv.py** - CSV to documents converter
8. **setup_nltk.py** - NLTK data downloader

### Data Files

- **data/documents/** - 2,692 converted news articles
- **data/news_queries.json** - Evaluation queries with ground truth
- **index/** - Saved index files (inverted index, TF-IDF matrix)
- **results/** - Evaluation results

### Performance Statistics

- **Index Building Time**: ~1 minute
- **Query Response Time**: <1 second
- **Index Size**: Compact (saved as pickle files)
- **Memory Efficient**: Sparse matrix representation

### Assignment Requirements Met

✅ **Local Implementation** - 100% local, no cloud services
✅ **Retrieval Strategy** - Multiple strategies (TF-IDF, BM25, Boolean)
✅ **Complete System** - End-to-end functional
✅ **Real Dataset** - 2,692 business news articles
✅ **Well-Evaluated** - Multiple metrics (P, R, F1, MAP, MRR, NDCG)
✅ **Reproducible** - Clear setup and usage instructions
✅ **Documented** - Comprehensive README and comments

---

## Next Steps

You can now:
1. Run more queries to test the system
2. Add more documents to expand the collection
3. Tune preprocessing parameters in config.py
4. Compare TF-IDF vs BM25 performance
5. Create additional evaluation queries
6. Experiment with different retrieval strategies

The system is ready for your information retrieval assignment submission!
