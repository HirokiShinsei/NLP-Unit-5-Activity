 Natural Language Processing with Wikipedia Corpus

This project demonstrates basic Natural Language Processing (NLP) techniques by fetching Wikipedia articles on astronomy topics and processing them using two approaches:

- **TF-IDF and Term Frequency**  
  Creates a term-document matrix using raw frequency and TF-IDF weighted approaches.

- **Word2Vec and Document Classification**  
  Uses Word2Vec to create dense document embeddings and trains a Logistic Regression classifier on these embeddings.

## Repository Structure

## Requirements

Ensure you have Python installed along with the following packages:

- `wikipedia`
- `numpy`
- `gensim`
- `scikit-learn`
- `re` (standard library)

You can install the required packages using pip:

```bat
pip install wikipedia numpy gensim scikit-learn
