from typing import List, Tuple
from .utils import normalize_text
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

class SimpleNearest:
    """Fallback simple nearest using set Jaccard similarity on tokens."""
    def __init__(self, docs: List[str]):
        self.docs = [normalize_text(d) for d in docs]
        self.tokens = [set(d.lower().split()) for d in self.docs]

    def query(self, q: str, topk: int = 1) -> List[Tuple[int, float]]:
        qn = normalize_text(q).lower().split()
        qset = set(qn)
        scores = []
        for i, toks in enumerate(self.tokens):
            inter = len(qset & toks)
            union = len(qset | toks) or 1
            scores.append((i, inter/union))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

class TfidfSearch:
    def __init__(self, docs: List[str]):
        self.docs = [normalize_text(d) for d in docs]
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(self.docs)
        else:
            # Minimal fallback: store docs and use SimpleNearest
            self.fallback = SimpleNearest(docs)

    def query(self, q: str, topk: int = 1):
        qn = normalize_text(q)
        if SKLEARN_AVAILABLE:
            qv = self.vectorizer.transform([qn])
            sims = cosine_similarity(qv, self.matrix)[0]
            import numpy as np
            idxs = list(np.argsort(-sims)[:topk])
            return [(int(i), float(sims[i])) for i in idxs]
        else:
            return self.fallback.query(q, topk)
