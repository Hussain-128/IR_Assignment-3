"""
Setup script to download required NLTK data
"""
import nltk


def download_nltk_data():
    """Download required NLTK datasets"""
    print("Downloading NLTK data...")
    
    required_data = [
        'punkt',           # Tokenizer
        'stopwords',       # Stopwords
        'wordnet',         # WordNet lemmatizer
        'averaged_perceptron_tagger',  # POS tagger
        'omw-1.4'         # Open Multilingual Wordnet
    ]
    
    for data in required_data:
        try:
            print(f"Downloading '{data}'...")
            nltk.download(data, quiet=False)
            print(f"✓ '{data}' downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading '{data}': {e}")
    
    print("\nNLTK data download complete!")


if __name__ == "__main__":
    download_nltk_data()
