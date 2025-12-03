

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

Top 10 Results:

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

