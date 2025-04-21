import re
import wikipedia
from term_frequency import compute_tf
from tf_idf import compute_idf, compute_tfidf

def tokenize(text):
    # Tokenization
    return re.findall(r'\w+', text.lower())

def main():
    # Documents to be pulled from the wikipedia corpus
    topics = [ "Black hole", "Supernova", "Galaxy", "Exoplanet", "Nebula" ]
    
    # store tokenized documents
    tokenized_docs = {}
    
    # Fetch and tokenize content for each topic.
    for topic in topics:
        print(f"Fetching Wikipedia article for: {topic}")
        try:
            page = wikipedia.page(topic)
            text = page.content
        except Exception as e:
            text = ""
            print(f"Error fetching {topic}: {e}")
        tokens = tokenize(text)
        tokenized_docs[topic] = tokens

    # Global vocabularo for all documents.
    vocab = set()
    for tokens in tokenized_docs.values():
        vocab.update(tokens)
    vocab = sorted(list(vocab)) 

    # Term-Document Matrix using raw term frequency.
    term_matrix = {}
    for topic, tokens in tokenized_docs.items():
        tf_vector = compute_tf(tokens, vocab)
        term_matrix[topic] = tf_vector

    # Compute IDF for the vocabulary of all documents.
    tokenized_docs_list = list(tokenized_docs.values())
    idf = compute_idf(tokenized_docs_list, vocab)

    # TF-IDF Matrix.
    tfidf_matrix = {}
    for topic, tokens in tokenized_docs.items():
        tf_vector = compute_tf(tokens, vocab)
        tfidf_vector = compute_tfidf(tf_vector, idf, vocab)
        tfidf_matrix[topic] = tfidf_vector

    # Results
    print("\nTerm-Document Matrix (Raw Frequency):")
    for topic in topics:
        print(f"\nTopic: {topic}")
        sample_tf = dict(list(term_matrix[topic].items())[:5])
        print(sample_tf)

    print("\nTerm-Document Matrix (TF-IDF):")
    for topic in topics:
        print(f"\nTopic: {topic}")
        sample_tfidf = dict(list(tfidf_matrix[topic].items())[:5])
        print(sample_tfidf)

if __name__ == "__main__":
    main()