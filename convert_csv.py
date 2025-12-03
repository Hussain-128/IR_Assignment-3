"""
CSV to Documents Converter
Converts Articles.csv into individual text documents for IR system
"""
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def convert_csv_to_documents(csv_file, output_dir):
    """Convert CSV articles to individual text documents"""
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            df = pd.read_csv(csv_file, encoding=encoding)
            print(f"✓ Successfully loaded with {encoding} encoding")
            break
        except Exception as e:
            print(f"✗ Failed with {encoding}: {str(e)[:50]}")
            continue
    
    if df is None:
        print("Failed to load CSV with any encoding")
        return 0
    
    print(f"\nTotal articles: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each article to a text file
    count = 0
    for idx, row in df.iterrows():
        try:
            # Create filename from date and heading
            date = str(row['Date']).replace('/', '-')
            heading = str(row['Heading'])[:50]  # Limit filename length
            # Clean filename
            filename = f"article_{idx+1:04d}_{date}.txt"
            filename = filename.replace(' ', '_').replace('/', '_')
            
            # Create document content
            content = f"Title: {row['Heading']}\n"
            content += f"Date: {row['Date']}\n"
            content += f"Category: {row['NewsType']}\n"
            content += f"\n{row['Article']}\n"
            
            # Write to file
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            count += 1
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"\n✓ Successfully converted {count} articles to text documents")
    print(f"✓ Documents saved in: {output_dir}")
    return count


if __name__ == "__main__":
    csv_file = "d:\\ASSSS\\Articles.csv"
    output_dir = "d:\\ASSSS\\data\\documents"
    
    # Clear existing sample documents
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.txt'):
                os.remove(os.path.join(output_dir, file))
        print("Cleared existing documents\n")
    
    convert_csv_to_documents(csv_file, output_dir)
