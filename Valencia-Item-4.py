import re
import wikipedia
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def tokenize(text):
    # Normalization and tokenization
    return re.findall(r'\w+', text.lower())

def fetch_documents(topics):
    # Fetch and tokenize documents from Wikipedia
    tokenized_docs = {}
    for topic in topics:
        print(f"Fetching Wikipedia article for: {topic}")
        try:
            page = wikipedia.page(topic)
            text = page.content
        except Exception as e:
            text = ""
            print(f"Error fetching '{topic}': {e}")
        tokens = tokenize(text)
        tokenized_docs[topic] = tokens
    return tokenized_docs

def train_word2vec(tokenized_docs, vector_size=100, window=5, min_count=2):
    
    # Combine all documents tokens
    sentences = list(tokenized_docs.values())
    model = Word2Vec(sentences, vector_size=vector_size, window=window, 
                     min_count=min_count, workers=4, sg=1)
    return model

def document_vector(tokens, model):
   
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        # Return a zero vector if no token is found in the model's vocabulary.
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

def main():
    # Topics
    topics = ["Black hole", "Supernova", "Galaxy", "Exoplanet", "Nebula"]
    
    # Fetch and tokenize documents
    tokenized_docs = fetch_documents(topics)
    
    # Train Word2Vec model on the tokens from all documents
    w2v_model = train_word2vec(tokenized_docs)
    
    # Build document embeddings by averaging word embeddings for each document.
    doc_embeddings = []
    labels = []
    for topic, tokens in tokenized_docs.items():
        vec = document_vector(tokens, w2v_model)
        doc_embeddings.append(vec)
        labels.append(topic)
    
    doc_embeddings = np.array(doc_embeddings)
    
    # Encode labels to numerical values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Configure Logistic Regression with L2 regularization.
    # Adjusting to prevent overfitting on small dataset.
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial',
                             solver='newton-cg', penalty='l2', C=0.1)
    clf.fit(doc_embeddings, y)
    
    # Predict on training data (note: dataset is very small)
    y_pred = clf.predict(doc_embeddings)
    
    # Output classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    print("\nTrained Logistic Regression classifier on dense document embeddings with regularization.")

if __name__ == "__main__":
    main()