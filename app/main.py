from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from .schemas import AnswerRequest, AnswerResponse
from .utils import normalize_text
from .tfidf_search import TfidfSearch, SimpleNearest
from typing import Dict
from starlette.responses import JSONResponse

app = FastAPI(title="FastAPI IA Challenge")

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory rate limit (very simple): {ip: [timestamps]}
RATE_LIMIT = 10  # requests
RATE_PERIOD = 60  # seconds
ip_reqs: Dict[str, list] = {}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    client = request.client.host if request.client else 'unknown'
    # rate limiting
    now = time.time()
    timestamps = ip_reqs.get(client, [])
    # drop old
    timestamps = [t for t in timestamps if now - t < RATE_PERIOD]
    if len(timestamps) >= RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail":"rate limit exceeded"})
    timestamps.append(now)
    ip_reqs[client] = timestamps
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("Error handling request")
        raise
    process_time = time.time() - start
    logger.info(f"{request.method} {request.url.path} from={client} time={process_time:.3f}s status={response.status_code}")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Example small knowledge base (could be replaced by artifacts/train)
DEFAULT_DOCS = [
    "{Paulo Gonçalves] Data Entrega 03-09-2025",
]

# Initialize searcher
searcher = TfidfSearch(DEFAULT_DOCS)

@app.post("/v1/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    start = time.time()
    q = req.question
    ctx = req.context
    if len(ctx) < 400:
        strategy = "tfidf"
        # Use TF-IDF + cosine
        hits = searcher.query(q + " " + ctx, topk=1)
        if hits:
            idx, score = hits[0]
            doc = searcher.docs[idx] if hasattr(searcher, 'docs') else DEFAULT_DOCS[idx]
            # Very simple extraction: if dates look like dd/mm or dd/mm/yyyy, return them
            import re
            dates = re.findall(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", doc)
            answer = doc if dates else (doc[:200] + "...")
        else:
            answer = "Não encontrei informação relevante."
    else:
        strategy = "alternative"
        # Simple fallback strategy (lightweight)
        from .tfidf_search import SimpleNearest
        fallback = SimpleNearest([ctx])
        hits = fallback.query(q, topk=1)
        answer = ctx if hits and hits[0][1] > 0 else "Não encontrei informação relevante."

    elapsed = time.time() - start
    logger.info(f"answered in {elapsed:.3f}s strategy={strategy}")
    return {"answer": answer, "strategy": strategy}
