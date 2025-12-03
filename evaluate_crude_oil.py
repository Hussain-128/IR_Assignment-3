"""
Manual Evaluation for Query: "crude oil"
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load CSV
df = pd.read_csv('Articles.csv', encoding='latin-1')

# Find all documents containing "crude oil"
crude_docs = df[df['Article'].str.contains('crude oil', case=False, na=False)]

print("=" * 70)
print("QUERY: 'crude oil'")
print("=" * 70)

print(f"\n✓ Total documents in dataset: {len(df)}")
print(f"✓ Total documents containing 'crude oil': {len(crude_docs)}")

# Create document IDs with dates to match our file naming
relevant_doc_ids = []
for idx in crude_docs.index:
    date = str(df.iloc[idx]['Date']).replace('/', '-')
    doc_id = f"article_{idx+1:04d}_{date}"
    relevant_doc_ids.append(doc_id)

print(f"\n--- RELEVANT DOCUMENTS (Ground Truth) ---")
print(f"Total relevant documents: {len(relevant_doc_ids)}\n")
for i, doc_id in enumerate(relevant_doc_ids[:20], 1):
    idx = int(doc_id.split('_')[1]) - 1
    date = df.iloc[idx]['Date']
    heading = df.iloc[idx]['Heading'][:60]
    print(f"{i}. {doc_id} | {heading}...")

if len(relevant_doc_ids) > 20:
    print(f"... and {len(relevant_doc_ids) - 20} more documents")

# Retrieved documents from our system
retrieved = [
    'article_2554_12-21-2016',
    'article_2661_3-13-2017',
    'article_2679_3-13-2017',
    'article_0991_9-14-2016',
    'article_0713_6-9-2016',
    'article_0483_2-20-2016',
    'article_0525_3-15-2016',
    'article_0056_2-19-2015',
    'article_2575_12-30-2016',
    'article_0627_5-12-2016'
]

# Extract just the base document ID (without date suffix for comparison)
retrieved_base = [doc.rsplit('_', 2)[0] + '_' + doc.rsplit('_', 2)[1] for doc in retrieved]

print(f"\n--- RETRIEVED DOCUMENTS (System Output) ---")
print(f"Top-10 retrieved documents:\n")
for i, doc in enumerate(retrieved, 1):
    print(f"{i}. {doc}")

# Calculate Precision and Recall
retrieved_set = set(retrieved)
relevant_set = set(relevant_doc_ids)

# Find intersection
true_positives = retrieved_set.intersection(relevant_set)

precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0
recall = len(true_positives) / len(relevant_set) if relevant_set else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'=' * 70}")
print("EVALUATION METRICS")
print("=" * 70)

print(f"\nTrue Positives (Relevant & Retrieved): {len(true_positives)}")
print(f"False Positives (Irrelevant but Retrieved): {len(retrieved_set) - len(true_positives)}")
print(f"False Negatives (Relevant but Not Retrieved): {len(relevant_set) - len(true_positives)}")

print(f"\n{'─' * 70}")
print(f"Precision: {precision:.4f} ({len(true_positives)}/{len(retrieved_set)})")
print(f"  → Out of 10 retrieved documents, {len(true_positives)} are relevant")
print(f"\nRecall:    {recall:.4f} ({len(true_positives)}/{len(relevant_set)})")
print(f"  → Out of {len(relevant_set)} relevant documents, {len(true_positives)} were retrieved")
print(f"\nF1 Score:  {f1:.4f}")
print(f"  → Harmonic mean of Precision and Recall")
print("=" * 70)

if len(true_positives) > 0:
    print(f"\n✓ Correctly Retrieved Documents:")
    for doc in true_positives:
        print(f"  • {doc}")
