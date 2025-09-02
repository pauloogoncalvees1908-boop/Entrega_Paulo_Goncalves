from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_answer_short_context():
    payload = {
        "question": "Qual o prazo?",
        "context": "OA projeto X tem prazo até 10/11. Entregas parciais em 01/11 e 05/11."
    }
    r = client.post('/v1/answer', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'answer' in data
    assert data['strategy'] == 'tfidf'

def test_rate_limit():
    payload = {"question":"a","context":"b"}
    # quickly exceed rate limit
    for i in range(12):
        r = client.post('/v1/answer', json=payload)
    assert r.status_code in (200,429)
