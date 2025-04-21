from collections import defaultdict

class InvertedIndex:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(set)
        self.vocabulary = set()
        
        for doc_id, text in self.documents.items():
            words = text.lower().split()
            self.vocabulary.update(words)
            for word in words:
                self.index[word].add(doc_id)

    def search(self, query):
        corrected_words = [self.correct_spelling(word) for word in query.lower().split()]
        print(f"Corrected Query: {' '.join(corrected_words)}")
        
        result_sets = [self.index[word] for word in corrected_words if word in self.index]
        return set.intersection(*result_sets) if result_sets else set()

    def correct_spelling(self, word):
        if word in self.vocabulary:
            return word
        return min(self.vocabulary, key=lambda vocab_word: self.levenshtein_distance(word, vocab_word))

    def levenshtein_distance(self, s1, s2):
        dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
        
        for i in range(len(s1) + 1): dp[i][0] = i
        for j in range(len(s2) + 1): dp[0][j] = j
            
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
                
        return dp[len(s1)][len(s2)]

documents = {
   1: "Web content extraction involves retrieving structured data",
   2: "Search engines use document indexing for efficient retrieval",
   3: "Document retrieval is important in web mining applications",
   4: "Indexing helps in retrieving relevant documents based on query terms"
}

index = InvertedIndex(documents)
queries = ["retrievel", "documnt indexing", "web minng", "strctured data"]

print("\n=== Spelling Correction and Document Retrieval ===")
for query in queries:
   result = index.search(query)
   print(f"Query: '{query}' → Corrected Documents: {sorted(result) if result else 'No matching documents'}")