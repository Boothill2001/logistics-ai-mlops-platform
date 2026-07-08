"""Local lexical embedding for Chroma.

Uses sklearn's HashingVectorizer (word 1-2 grams, l2-normalized) — stateless
and deterministic: no fitted vocabulary to persist, ingest and query always
embed identically, zero model download.

Trade-off (deliberate for this demo): this is a *lexical* embedding — it
matches on shared terms, not meaning ("vessel late" won't match "ship
delayed"). Production swap: any sentence-embedding model behind this same
interface; nothing else in the codebase changes.
"""
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 512


class LocalHashEmbedding(EmbeddingFunction):
    def __init__(self):
        self._vectorizer = HashingVectorizer(
            n_features=N_FEATURES,
            ngram_range=(1, 2),
            norm="l2",
            alternate_sign=False,
            lowercase=True,
        )

    def __call__(self, input: Documents) -> Embeddings:
        matrix = self._vectorizer.transform(input)
        return matrix.toarray().tolist()

    @staticmethod
    def name() -> str:
        return "local_hash_embedding"
