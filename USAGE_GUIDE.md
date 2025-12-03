# How to Use the Interactive Search with Evaluation

## Step-by-Step Guide

### Step 1: Start Interactive Mode
```bash
py -3.11 main.py --interactive
```

### Step 2: Enter Your Query
When prompted, enter your search query, for example:
```
Enter your query: crude oil
```

### Step 3: View Search Results
The system will display the top 10 matching documents with:
- Document ID
- Relevance Score
- File name
- Document snippet

Example output:
```
Found 10 results for query: 'crude oil'
======================================================================

1. Document: article_2554_12-21-2016
   Score: 0.2721
   File: article_2554_12-21-2016.txt
   Snippet: Oil prices edge up on expected US crude inventory...
   
2. Document: article_2661_3-13-2017
   Score: 0.2519
   ...
```

### Step 4: Choose Evaluation Option
After results are displayed, you'll see:
```
======================================================================
OPTIONS:
  1. Evaluate this query (Calculate Precision, Recall, etc.)
  2. Exit
======================================================================

Enter your choice (1 or 2):
```

**Choose 1** to evaluate the query
**Choose 2** to exit the program

### Step 5: Enter Relevant Document IDs (if you chose 1)
The system will ask for relevant document IDs:
```
----------------------------------------------------------------------
Please enter the relevant document IDs for this query
(comma-separated, e.g., article_0001_1-1-2015, article_0002_1-2-2015)
Hint: Look at the document IDs shown in the search results above
----------------------------------------------------------------------

Relevant document IDs:
```

Enter the document IDs that you consider relevant, separated by commas:
```
article_2554_12-21-2016, article_2661_3-13-2017, article_0056_2-19-2015, article_0013_1-16-2015
```

### Step 6: View Evaluation Results
The system will calculate and display:

```
======================================================================
EVALUATION RESULTS
======================================================================

Query: 'crude oil'
Retrieved documents: 10
Relevant documents (provided): 4

True Positives (Relevant & Retrieved): 3
False Positives (Retrieved but Not Relevant): 7
False Negatives (Relevant but Not Retrieved): 1

----------------------------------------------------------------------
Precision: 0.3000 (3/10)
  → 30.00% of retrieved documents are relevant

Recall:    0.7500 (3/4)
  → 75.00% of relevant documents were retrieved

F1 Score:  0.4286
  → Harmonic mean of Precision and Recall
======================================================================

✓ Correctly Retrieved Documents:
  • article_0056_2-19-2015
  • article_2554_12-21-2016
  • article_2661_3-13-2017
```

## Example Session

```bash
# Start the system
py -3.11 main.py --interactive

# Enter query
Enter your query: crude oil

# View results (top 10 documents shown)

# Choose evaluation
Enter your choice (1 or 2): 1

# Enter relevant documents
Relevant document IDs: article_2554_12-21-2016, article_2661_3-13-2017, article_0056_2-19-2015

# View evaluation metrics
Precision: 0.3000
Recall: 0.7500
F1 Score: 0.4286
```

## Tips

1. **Finding Relevant Documents**: Look at the search results and identify which documents truly answer your query
2. **Document ID Format**: Always use the format `article_XXXX_M-D-YYYY` (e.g., `article_0056_2-19-2015`)
3. **Multiple Documents**: Separate document IDs with commas: `doc1, doc2, doc3`
4. **Exit Anytime**: Choose option 2 to exit the program

## Understanding Metrics

- **Precision**: What percentage of retrieved documents are actually relevant?
  - High precision = Few irrelevant documents in results
  
- **Recall**: What percentage of relevant documents were found?
  - High recall = Most relevant documents were retrieved
  
- **F1 Score**: Balance between Precision and Recall
  - Closer to 1.0 = Better overall performance

## Example Queries to Try

1. `crude oil`
2. `stock market karachi`
3. `electricity load shedding`
4. `petrol prices pakistan`
5. `IMF loan economy`
6. `dollar rupee exchange`
