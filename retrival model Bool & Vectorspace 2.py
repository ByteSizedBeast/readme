from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class BooleanRetrieval:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(set)
        for doc_id, text in documents.items():
            for word in set(text.lower().split()):
                self.index[word].add(doc_id)
    
    def search(self, query):
        terms = query.lower().split()
        result = set(self.documents.keys())
        op = "AND"
        
        for term in terms:
            if term in ["and", "or", "not"]:
                op = term.upper()
            else:
                term_docs = self.index.get(term, set())
                if op == "AND":
                    result &= term_docs
                elif op == "OR":
                    result |= term_docs
                elif op == "NOT":
                    result -= term_docs
        return result

class VectorRetrieval:
    def __init__(self, documents):
        self.doc_ids = list(documents.keys())
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(documents.values())
    
    def search(self, query):
        query_vector = self.vectorizer.transform([query])
        scores = np.dot(self.doc_vectors, query_vector.T).toarray().flatten()
        return [(self.doc_ids[i], round(scores[i], 4)) 
                for i in range(len(self.doc_ids)) if scores[i] > 0]

docs = {
    1: "Web content extraction involves retrieving structured data",
    2: "Search engines use document indexing for efficient retrieval",
    3: "Document retrieval is important in web mining applications",
    4: "Indexing helps in retrieving relevant documents based on query terms"
}

print("\n=== Boolean Retrieval ===")
boolean_engine = BooleanRetrieval(docs)
for query in ["retrieval AND document", "document OR indexing", "retrieval NOT indexing"]:
    result = boolean_engine.search(query)
    print(f"Query: '{query}' → Docs: {sorted(result) or 'None'}")

print("\n=== Vector Space Retrieval ===")
vector_engine = VectorRetrieval(docs)
for query in ["document retrieval", "web mining", "structured data"]:
    result = vector_engine.search(query)
    print(f"Query: '{query}' → Ranked: {sorted(result, key=lambda x: x[1], reverse=True)}")