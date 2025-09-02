"""train.py
Script simples que treina TF-IDF sobre faq.csv (question,answer) e salva artefatos em artifacts/.
Gera métricas simples (MRR).
"""
import csv, os, pickle
from pathlib import Path
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN = True
except Exception:
    SKLEARN = False

ROOT = Path(__file__).parent
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)

faq_path = ROOT / 'faq.csv'
if not faq_path.exists():
    print('faq.csv not found. Create a CSV with headers: question,answer')
    exit(0)

qas = []
with open(faq_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        qas.append((row[0], row[1] if len(row)>1 else ''))

questions = [q for q,_ in qas]
answers = [a for _,a in qas]

if SKLEARN:
    vec = TfidfVectorizer()
    mat = vec.fit_transform(answers)
    # save
    with open(ART/'vectorizer.pkl','wb') as f:
        pickle.dump(vec, f)
    with open(ART/'matrix.pkl','wb') as f:
        pickle.dump(mat, f)
    print('Saved artifacts in', ART)
else:
    print('Scikit-learn not available; nothing saved.')

# Metrics: naive MRR on answers-as-candidates
def mrr():
    if not SKLEARN: return
    import numpy as np
    ranks = []
    for qi, (q,a) in enumerate(qas):
        qv = vec.transform([q])
        sims = cosine_similarity(qv, mat)[0]
        order = (-sims).argsort()
        rank = list(order).index(qi) + 1 if qi < len(order) else None
        if rank:
            ranks.append(1.0/rank)
    if ranks:
        print('MRR:', sum(ranks)/len(ranks))
mrr()
