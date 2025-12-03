

### Required Software

- **Python 3.11** (Recommended - Stable with all dependencies)
- **pip** (Python package installer)
- **Git** (For version control)
- **4 GB RAM** (Minimum)
- **500 MB Disk Space** (For dataset and indices)


##  Installation & Setup

### Step 1: Install Python 3.11

#### Windows:
```powershell
# Download Python 3.11 from python.org
# Or use Windows Store
# Verify installation
py -3.11 --version
```

# Verify installation
python3.11 --version
```

---

### Step 2: Clone/Download the Project

```bash
# Navigate to your directory
cd "Your directory"

---

### Step 3: Create Virtual Environment (Recommended)

```powershell
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# You should see (venv) in your prompt
```

```bash
# macOS/Linux
python3.11 -m venv venv
source venv/bin/activate

# You should see (venv) in your prompt
```

---

### Step 4: Install Dependencies

```powershell
# Upgrade pip first
py -3.11 -m pip install --upgrade pip

# Install required packages
py -3.11 -m pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 nltk==3.8.1 rank-bm25==0.2.2 tqdm==4.66.1

# Or install from requirements.txt (if provided)
py -3.11 -m pip install -r requirements.txt
```

**Expected packages:**
- `pandas==2.0.3` - Data handling
- `numpy==1.24.3` - Numerical operations
- `scikit-learn==1.3.0` - TF-IDF vectorization
- `nltk==3.8.1` - Natural language processing
- `rank-bm25==0.2.2` - BM25 implementation
- `tqdm==4.66.1` - Progress bars

---

### Step 5: Download NLTK Data

```powershell
# Run the NLTK setup script
py -3.11 setup_nltk.py

# This will download:
# - punkt (tokenizer)
# - stopwords (English stopwords)
# - wordnet (lemmatization)
# - averaged_perceptron_tagger (POS tagging)
# - omw-1.4 (multilingual wordnet)
```

---

### Step 6: Place Dataset

Ensure `Articles.csv` is in the project root directory:

```
D:\ASSSS\
├── Articles.csv          ← Your dataset here
├── config.py
├── setup_nltk.py
├── main.py
└── ...
```

---

### Step 7: Run Complete Setup (Automated)

```powershell
# Run the automated setup script
py -3.11 setup.py

# This will:
# 1. Check Python version
# 2. Create directories
# 3. Install dependencies
# 4. Download NLTK data
# 5. Verify installation
```

---

##  Quick Start

### 1. Build the Search Index (First Time Only)

```powershell
py -3.11 main.py --build-index
```

**What this does:**
- Loads Articles.csv
- Preprocesses all documents (2,692 articles)
- Builds TF-IDF and BM25 indices
- Saves indices to disk (~2-3 minutes)

---

### 2. Start Interactive Search

```powershell
py -3.11 main.py --interactive
```

**Example Session:**
```
Enter your query: crude oil prices

📊 Top 10 Results:

[1] Score: 0.3247
    ID: article_0156_5-21-2015
    Title: Oil prices increase due to Yemen turmoil

OPTIONS:
  1. Evaluate this query
  2. New search
  3. Back to main menu
```

---

### 3. Run a Single Query

```powershell
# Search using hybrid method (TF-IDF + BM25)
py -3.11 main.py --query "stock market karachi" --top-k 5

# Search using only TF-IDF
py -3.11 main.py --query "stock market karachi" --method tfidf

# Search using only BM25
py -3.11 main.py --query "stock market karachi" --method bm25
```

---

##  Usage Guide

### Command-Line Options

```powershell
# Main menu (interactive mode)
py -3.11 main.py

# Build/rebuild index
py -3.11 main.py --build-index

# Interactive search with evaluation
py -3.11 main.py --interactive

# Single query search
py -3.11 main.py --query "your search query"

# Specify retrieval method
py -3.11 main.py --query "your query" --method [tfidf|bm25|hybrid]

# Set number of results
py -3.11 main.py --query "your query" --top-k 20

# View configuration
py -3.11 config.py

# Verify NLTK setup
py -3.11 setup_nltk.py --verify

# View help
py -3.11 main.py --help
```

---

## 📁 Project Structure

```
D:\ASSSS\
│
├── Articles.csv                 # Dataset (2,692 news articles)
│
├── config.py                    # System configuration
├── setup_nltk.py                # NLTK data downloader
├── setup.py                     # Automated setup script
├── main.py                      # Main entry point with menu
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── preprocessing.py             # Text preprocessing module
├── indexer.py                   # Indexing (TF-IDF + BM25)
├── retrieval.py                 # Search and ranking
├── evaluation.py                # Evaluation metrics
│
├── data/                        # Data directory
│   └── processed/               # Preprocessed documents
│
├── models/                      # Saved indices
│   ├── tfidf_model.pkl         # TF-IDF vectorizer
│   ├── tfidf_matrix.pkl        # TF-IDF document matrix
│   ├── bm25_model.pkl          # BM25 index
│   └── inverted_index.pkl      # Inverted index
│
└── results/                     # Search results and logs
    └── evaluation_results.json
```

---

## ⚙️ Configuration

Edit `config.py` to customize system behavior:

### Preprocessing Settings

```python
PREPROCESSING = {
    'lowercase': True,              # Convert to lowercase
    'remove_stopwords': True,       # Remove common words
    'stemming': True,               # Apply Porter Stemmer
    'min_word_length': 2,           # Minimum word length
}
```

### TF-IDF Parameters

```python
TFIDF_PARAMS = {
    'ngram_range': (1, 2),          # Unigrams and bigrams
    'min_df': 2,                    # Min document frequency
    'max_df': 0.85,                 # Max document frequency
    'sublinear_tf': True,           # Use log(tf) + 1
    'norm': 'l2',                   # L2 normalization
}
```

### BM25 Parameters

```python
BM25_PARAMS = {
    'k1': 1.5,                      # Term frequency saturation
    'b': 0.75,                      # Length normalization
}
```

### Hybrid Weights

```python
RETRIEVAL = {
    'top_k': 10,
    'hybrid_weights': {
        'tfidf': 0.5,               # 50% TF-IDF
        'bm25': 0.5,                # 50% BM25
    }
}
```

---

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Precision
```
Precision = (Relevant Documents Retrieved) / (Total Documents Retrieved)
```

### Recall
```
Recall = (Relevant Documents Retrieved) / (Total Relevant Documents)
```

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Mean Average Precision (MAP)
Average precision across multiple queries.

### Mean Reciprocal Rank (MRR)
Average of reciprocal ranks of first relevant document.

### Normalized Discounted Cumulative Gain (NDCG)
Measures ranking quality with graded relevance.

---

## 🔧 Troubleshooting

### Issue: "No module named 'nltk'"
**Solution:**
```powershell
py -3.11 -m pip install nltk
```

### Issue: "NLTK data not found"
**Solution:**
```powershell
py -3.11 setup_nltk.py
```

### Issue: "Articles.csv not found"
**Solution:**
Ensure `Articles.csv` is in the project root directory `D:\ASSSS\`

### Issue: "Memory error during indexing"
**Solution:**
- Close other applications
- Use 64-bit Python 3.11

### Issue: "Python 3.11 not found"
**Solution:**
```powershell
# Download and install from python.org
# Verify with:
py -3.11 --version
```

### Issue: "Index files corrupted"
**Solution:**
```powershell
# Rebuild index
py -3.11 main.py --build-index
```

---

## 🔐 Git Setup (Second Account)

### Configure Second Git Account Locally

```powershell
# Navigate to your project
cd D:\ASSSS

# Initialize Git repository (if not already done)
git init

# Configure local Git settings for this project only
git config user.name "Your Second Account Name"
git config user.email "your.second.email@example.com"

# Verify configuration
git config user.name
git config user.email
```

### Create .gitignore File

```powershell
# Create .gitignore to exclude large files
@"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/

# Data and Models
models/*.pkl
data/processed/
*.csv

# Results
results/
logs/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath .gitignore -Encoding utf8
```

### Initial Commit

```powershell
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Hybrid IR System with TF-IDF and BM25

- Implemented text preprocessing pipeline
- Built TF-IDF and BM25 indexing
- Created interactive search interface
- Added comprehensive evaluation metrics
- Configured for Python 3.11"

# View commit history
git log --oneline
```

### Connect to Remote Repository (GitHub)

```powershell
# Create repository on GitHub first with your second account
# Then connect it:

git remote add origin https://github.com/YOUR_SECOND_USERNAME/IR_System.git

# Or using SSH (recommended):
git remote add origin git@github.com:YOUR_SECOND_USERNAME/IR_System.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Using Personal Access Token (PAT) for Second Account

```powershell
# When prompted for password, use Personal Access Token instead

# Generate PAT on GitHub:
# 1. Go to GitHub → Settings → Developer settings → Personal access tokens
# 2. Generate new token (classic)
# 3. Select scopes: repo (full control)
# 4. Copy the token

# When pushing:
git push -u origin main
# Username: YOUR_SECOND_USERNAME
# Password: paste_your_PAT_here

# Save credentials (optional):
git config credential.helper store
```

### SSH Key Setup (Recommended for Second Account)

```powershell
# Generate new SSH key for second account
ssh-keygen -t ed25519 -C "your.second.email@example.com" -f $HOME\.ssh\id_ed25519_second

# Start SSH agent
Start-Service ssh-agent

# Add key to agent
ssh-add $HOME\.ssh\id_ed25519_second

# Copy public key (add to GitHub)
Get-Content $HOME\.ssh\id_ed25519_second.pub | clip
```

Then add the public key to your second GitHub account:
- Go to GitHub → Settings → SSH and GPG keys → New SSH key
- Paste the public key

### Configure SSH Config (Multiple Accounts)

Create/edit `$HOME\.ssh\config`:

```
# Second GitHub Account
Host github-second
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_second
    IdentitiesOnly yes
```

Then use:
```powershell
git remote set-url origin git@github-second:YOUR_SECOND_USERNAME/IR_System.git
git push origin main
```

### Subsequent Commits

```powershell
# After making changes
git status                          # Check what changed
git add .                          # Stage all changes
git commit -m "Your commit message"  # Commit with message
git push origin main                # Push to GitHub
```

---

## 🎓 Example Queries to Try

```
1. "crude oil prices international market"
2. "stock market karachi exchange index"
3. "Pakistan IMF loan agreement"
4. "inflation rate economy growth"
5. "electricity power crisis load shedding"
6. "rupee dollar exchange rate"
7. "export import trade deficit"
8. "fiscal budget government revenue"
9. "banking sector profit growth"
10. "textile industry cotton prices"
```

---

## 📈 Expected Performance

- **Index Build Time**: ~2-3 minutes (2,692 documents)
- **Query Response Time**: < 200ms
- **Index Size**: ~20-30 MB
- **Memory Usage**: ~100-200 MB
- **Precision@10**: ~0.15-0.20 (typical for news search)
- **Recall@10**: ~0.35-0.45

---

## 📄 License

This project is created for academic purposes as part of an Information Retrieval course assignment.

---

## 👨‍💻 Author

**[Your Name]**
- Course: Information Retrieval
- Date: December 2025
- Institution: [Your University]

---

## 🙏 Acknowledgments

- Dataset: Pakistani Business News Articles (2015-2017)
- Libraries: scikit-learn, NLTK, rank-bm25
- Retrieval Models: TF-IDF, BM25 (Okapi)

---

**Last Updated**: December 3, 2025
**Version**: 1.0.0
**Python Version**: 3.11 (Required)

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  QUICK COMMAND REFERENCE                        │
├─────────────────────────────────────────────────┤
│  Setup:                                         │
│    py -3.11 setup.py                           │
│    py -3.11 setup_nltk.py                      │
│                                                 │
│  Build Index:                                   │
│    py -3.11 main.py --build-index              │
│                                                 │
│  Search:                                        │
│    py -3.11 main.py --interactive              │
│    py -3.11 main.py --query "text"             │
│                                                 │
│  Methods:                                       │
│    --method tfidf                              │
│    --method bm25                               │
│    --method hybrid (default)                   │
│                                                 │
│  Git Commands:                                  │
│    git add .                                   │
│    git commit -m "message"                     │
│    git push origin main                        │
│                                                 │
│  Help:                                          │
│    py -3.11 main.py --help                     │
└─────────────────────────────────────────────────┘
```
