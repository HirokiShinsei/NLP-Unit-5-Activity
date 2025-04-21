import re
import wikipedia
import itertools
from term_frequency import compute_tf
from cosine_similarity import cosine_similarity

def tokenize(text):
    # normalization and tokenization
    return re.findall(r'\w+', text.lower())

def main():
    # Define topics 
    topics = ["Black hole", "Supernova", "Galaxy", "Exoplanet", "Nebula"]
    
    # tokenized topics
    tokenized_docs = {}
    for topic in topics:
        print(f"Fetching Wikipedia article for: {topic}")
        try:
            page = wikipedia.page(topic)
            text = page.content
        except Exception as e:
            text = ""
            print(f"Error fetching '{topic}': {e}")
        tokenized_docs[topic] = tokenize(text)
    
    # Build a global vocabulary from all documents.
    vocab = set()
    for tokens in tokenized_docs.values():
        vocab.update(tokens)
    vocab = sorted(list(vocab))

    # Get term frequency vectors for each document.
    tf_vectors = {}
    for topic, tokens in tokenized_docs.items():
        tf_vectors[topic] = compute_tf(tokens, vocab)
    
    # Compare each document
    max_similarity = -1.0
    most_similar_pair = None

    for topic1, topic2 in itertools.combinations(topics, 2):
        similarity = cosine_similarity(tf_vectors[topic1], tf_vectors[topic2], vocab)
        print(f"Cosine similarity between '{topic1}' and '{topic2}': {similarity:.4f}")
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_pair = (topic1, topic2)
    
    if most_similar_pair:
        print(f"\nThe most similar documents are '{most_similar_pair[0]}' and '{most_similar_pair[1]}' with a cosine similarity of {max_similarity:.4f}")

if __name__ == "__main__":
    main()