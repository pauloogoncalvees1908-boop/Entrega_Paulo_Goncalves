## Estrutura
```
Entrega Paulo Gonçalves/
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── utils.py
│   └── tfidf_search.py
├── tests/
│   ├── test_normalize.py
│   └── test_api.py
├── train.py
├── requirements.txt
└── README.md
```

## Como rodar
1. Crie um virtualenv e instale dependências:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Rode a API:
   ```bash
   uvicorn app.main:app --reload
   ```

3. Exemplo de requisição:
   ```bash
   curl -X POST "http://127.0.0.1:8000/v1/answer" -H "Content-Type: application/json" -d '{"question":"Qual o prazo?","context":"OA projeto X tem prazo até 10/11. Entregas parciais em 01/11 e 05/11."}'
   ```

## Testes
```bash
pytest -q
```

## Escolha técnicas (resumo)

FastAPI: Escolhido por seu desempenho excepcional, suporte nativo para validação de dados com Pydantic e geração automática de documentação interativa (Swagger UI), o que agiliza o desenvolvimento e a comunicação com outros times.

Pydantic: Utilizado para garantir a validação de entrada e saída, tornando a API robusta e prevenindo erros causados por dados mal formatados.

Estratégia de NLP (TF-IDF vs. Alternativa):

TF-IDF para Contextos Curtos (< 400 caracteres): TF-IDF é uma solução simples, leve e eficiente para calcular a relevância de palavras. Para contextos pequenos, é mais do que suficiente e oferece uma boa baseline.

Estratégia Alternativa para Contextos Longos (>= 400 caracteres): Para contextos maiores, o TF-IDF pode perder a capacidade de capturar a semântica, já que o modelo de "saco de palavras" não considera a ordem ou o significado das palavras. Uma estratégia alternativa (como a busca por palavras-chave implementada) ou até mesmo o uso de embeddings (como de modelos Sentence-BERT, que seriam uma escolha mais robusta em um cenário real) se torna mais adequada.


## Bônus (train.py)
- `train.py` lê `faq.csv` (question,answer), treina TF-IDF e salva artefatos em `artifacts/` (vectorizer + matrix via pickle).
- Métricas básicas (mean reciprocal rank) são medidas no dataset (se disponível).

