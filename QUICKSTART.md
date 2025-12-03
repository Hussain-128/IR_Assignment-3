# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python setup_nltk.py
```

## Step 2: Create Sample Data (or add your own documents)

```bash
python main.py --create-samples
```

This creates:
- 5 sample documents in `data/documents/`
- Sample queries file in `data/sample_queries.json`

**OR** place your own `.txt` documents in `data/documents/` directory.

## Step 3: Build the Index

```bash
python main.py --build-index
```

This will:
- Process all documents in `data/documents/`
- Build inverted index and TF-IDF vectors
- Save index to `index/` directory

## Step 4: Search!

### Single query:
```bash
python main.py --query "information retrieval"
```

### Interactive mode:
```bash
python main.py --interactive
```

### Try different methods:
```bash
python main.py --query "search engines" --method bm25
python main.py --query "vector space" --method boolean
```

## Step 5: Evaluate (Optional)

```bash
python main.py --evaluate --queries-file data/sample_queries.json
```

## Configuration

Edit `config.py` to customize:
- Preprocessing options (stemming, stopwords)
- Retrieval parameters (top-k, TF-IDF settings)
- BM25 parameters (k1, b)

## Troubleshooting

**NLTK errors?** Run `python setup_nltk.py` again.

**No documents found?** Make sure `.txt` files are in `data/documents/`

**Index not found?** Run `python main.py --build-index` first.

## System Architecture

```
Query → Preprocessing → Retrieval Engine → Ranking → Results
                ↓
        Document Index (TF-IDF/Inverted)
```

## Example Output

```
Query: information retrieval
Method: TFIDF

Found 3 results:
1. Document: doc1_ir_basics
   Score: 0.8234
   Snippet: Information retrieval (IR) is the process of obtaining...

2. Document: doc3_vector_space_model
   Score: 0.6521
   ...
```
