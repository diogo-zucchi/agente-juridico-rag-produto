# Agente Jurídico Inteligente — RAG

TCC — Engenharia de Software — UTFPR Dois Vizinhos — 2026
Autor: Diogo Eduardo Ferrari Zucchi
Orientador: Prof. Dr. Francisco Carlos Souza

Sistema de consulta jurídica em linguagem natural. O usuário indexa documentos jurídicos e faz perguntas; o sistema recupera os trechos mais relevantes por busca semântica e gera uma resposta fundamentada citando as fontes, permitindo verificar a procedência de cada informação.

O sistema também inclui um **instrumento de comparação às cegas** entre modelos de linguagem, usado durante o desenvolvimento para fundamentar a escolha do modelo. Esse instrumento não faz parte do fluxo destinado ao usuário final.

---

## Modelos de linguagem

A geração das respostas usa o padrão de API compatível com a OpenAI, de modo que a troca de modelo se resume a alterar a URL base, a credencial e o nome do modelo. Três modelos estão integrados:

| Provedor | Modelo | Papel |
|----------|--------|-------|
| Maritaca | `sabia-4` | **Modelo da ferramenta** (produção) — brasileiro, com pré-treino em português e legislação jurídica. |
| DeepSeek | `deepseek-chat` | Apenas para o estudo comparativo às cegas. |
| OpenAI | `gpt-4o-mini` | Apenas para o estudo comparativo às cegas. |

A ferramenta opera com o **Sabiá**; DeepSeek e GPT foram integrados exclusivamente para fundamentar a escolha do modelo na etapa de validação.

---

## Arquitetura

- **Frontend:** página única em HTML, CSS e JavaScript (`frontend/index.html`), servida pela própria API em `/`.
- **Backend:** Python 3.13 + FastAPI, organizado em camadas (rotas → serviços → persistência).
- **Persistência:** PostgreSQL com a extensão pgvector — dados estruturados e vetores no mesmo banco.
- **Embeddings:** `text-embedding-3-small` da OpenAI (1536 dimensões), por API. Qualquer provedor com API compatível com a OpenAI serve (ver `EMBEDDING_*` na configuração).

Cada chunk guarda o texto e o seu vetor na mesma linha da tabela `chunks` (ver "Persistência dos vetores"). Textos de lei recebem tratamento próprio na fragmentação (ver "Fragmentação de textos legais").

---

## Pré-requisitos

- Python 3.13
- Docker Desktop (só para o banco local; ou use um banco gerenciado também no desenvolvimento)
- Chave da OpenAI (embeddings) e da Maritaca (respostas)

---

## Instalação

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows (PowerShell)
# source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

---

## Configuração

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo (substitua as chaves pelas suas):

```env
# ── Banco de dados ───────────────────────────────────────
DATABASE_URL=postgresql+psycopg://postgres:senha@127.0.0.1:5433/agente_juridico
POSTGRES_USER=postgres
POSTGRES_PASSWORD=senha
POSTGRES_DB=agente_juridico

# ── Modelo da ferramenta (produção) ──────────────────────
MARITACA_API_KEY=sua_chave_maritaca
MARITACA_MODEL=sabia-4

# ── Modelos do estudo comparativo (avaliação às cegas) ───
DEEPSEEK_API_KEY=sua_chave_deepseek
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

OPENAI_API_KEY=sua_chave_openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# ── Embeddings (API compatível com a OpenAI) ─────────────
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=          # vazio: usa a OPENAI_API_KEY
EMBEDDING_DIM=1536

# ── Acesso ───────────────────────────────────────────────
API_TOKEN=                  # vazio: API aberta (só para uso local)
CORS_ORIGINS=               # vazio: só a própria API

# ── Recuperação e fragmentação ───────────────────────────
TOP_K_RESULTS=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
ARTIGO_MAX_CHARS=4000

# ── Logs ─────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FILE=                   # vazio: só stdout
```

> Para o uso normal da ferramenta bastam `MARITACA_API_KEY` (respostas) e `OPENAI_API_KEY` (embeddings). A chave do DeepSeek só é necessária para o estudo comparativo às cegas.
>
> `EMBEDDING_MODEL` e `EMBEDDING_DIM` andam juntos e definem a coluna `vector(N)` do banco. Para trocar de modelo, recrie o banco e indexe os documentos de novo.

---

## Execução

1. Subir o banco de dados (porta 5433):

```bash
docker compose up -d postgres
```

2. Iniciar a API (dentro da pasta `backend/`). Ao iniciar, ela habilita o pgvector e cria as tabelas que faltarem:

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8001
```

3. Abrir a interface em `http://localhost:8001/` (a API serve o `index.html` na raiz).

Documentação interativa da API (Swagger): `http://localhost:8001/docs`.

> As portas 5433 e 8001 (e não as padrão 5432 e 8000) deixam o produto rodar ao mesmo tempo que o TCC.

> Alternativa: `docker compose up --build` sobe banco e backend juntos, sem precisar de Python local. O `docker-compose.yml` é só para desenvolvimento (reload, código montado, senha padrão).

---

## Hospedagem

Em produção são duas peças, sem Docker:

1. **Banco:** um Postgres gerenciado com pgvector, como [Neon](https://neon.tech) ou [Supabase](https://supabase.com). Copie a URL de conexão (`postgresql://...`); a API acrescenta o driver sozinha. As tabelas são criadas na primeira inicialização.
2. **API:** uma plataforma que rode Python a partir do repositório, como [Render](https://render.com) ou [Railway](https://railway.app).

**Caminho pronto: Neon + Render.**

1. No Neon, crie um projeto na região **AWS US East (N. Virginia)** e copie a connection string (`postgresql://...?sslmode=require`).
2. No Render, **New → Blueprint** e escolha este repositório: o `render.yaml` cria o serviço na região Virginia, a mesma do banco.
3. O Render pede `DATABASE_URL`, `MARITACA_API_KEY` e `OPENAI_API_KEY`; o `API_TOKEN` é gerado por ele (copie-o em **Environment** para usar a página).
4. Quando o deploy terminar, `https://<serviço>.onrender.com/health` deve responder `"database": true`.

O plano `free` do Render dorme após 15 minutos sem acesso e leva cerca de um minuto para acordar; para demonstrações, o plano pago mais simples evita isso.

Em outra plataforma, a configuração é esta:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Python:** 3.13 (arquivo `.python-version`; no Render, também pela variável `PYTHON_VERSION`)
   - **Health check:** `/health`
   - **Variáveis:** `DATABASE_URL`, `MARITACA_API_KEY`, `OPENAI_API_KEY` e `API_TOKEN` (um valor longo e aleatório)

Com `API_TOKEN` definido, as rotas `/api/v1` exigem o cabeçalho `X-API-Key`. A página pede o token na primeira chamada recusada e o guarda no navegador. A página e o `/health` continuam públicos.

**Uma instância só.** A indexação roda em segundo plano no próprio processo da API: o upload responde na hora, e o status do documento (`processando`, `concluido` ou `erro`) aparece em `GET /documentos`. Se o servidor reiniciar no meio, o documento fica com erro e basta enviar o arquivo de novo. Com mais de uma instância, isso precisaria de uma fila.

O `backend/Dockerfile` continua disponível para plataformas que preferem imagem (build a partir da raiz: `docker build -f backend/Dockerfile .`).

---

## Uso

- **Indexar um documento:** use a área de upload na interface e selecione um arquivo PDF, TXT ou HTML, informando um título (e, opcionalmente, um nível de dificuldade usado apenas no experimento).
- **Consultar:** digite a pergunta jurídica e clique em Consultar. A resposta vem acompanhada das fontes utilizadas e seus escores de similaridade.
- **Comparar modelos (estudo):** na aba de avaliação, a mesma pergunta é respondida por todos os modelos sobre o mesmo contexto; as respostas são embaralhadas e exibidas como Resposta A, B, C, sem identificar o modelo, para avaliação às cegas com nota de 1 a 5.

Indexação em lote (todos os arquivos de uma pasta):

```bash
python scripts/indexar_documentos.py --pasta ./data/raw --fonte-id 1
```

---

## Endpoints principais

Prefixo: `/api/v1`

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/consultar` | Consulta com um modelo (padrão: Maritaca). |
| `POST` | `/comparar` | Gera respostas de todos os modelos sobre o mesmo contexto (às cegas). |
| `POST` | `/avaliar` | Registra a nota (1 a 5) de uma resposta. |
| `GET`  | `/resultados` | Consolida as notas por modelo e por nível de dificuldade. |
| `GET`  | `/resultados/csv` | Exporta as avaliações em CSV. |
| `POST` | `/ingerir` | Recebe um documento e o indexa em segundo plano (responde `202`). |
| `GET`  | `/documentos` | Lista os documentos e o status da indexação de cada um. |
| `GET`  | `/historico` | Histórico de consultas. |

---

## Persistência dos vetores

Os vetores ficam na coluna `chunks.embedding` (`vector(1536)`, extensão pgvector), ao lado do texto do chunk:

- **Busca:** produto interno (operador `<#>`) sobre embeddings normalizados — equivalente à similaridade de cosseno.
- **Busca exata:** não há índice vetorial aproximado. Com poucos milhares de chunks a varredura é rápida e o resultado é determinístico. Se o volume crescer, crie um índice HNSW: `CREATE INDEX ON chunks USING hnsw (embedding vector_ip_ops);`
- **Filtro por documento:** feito na própria consulta SQL (`WHERE id_documento = ...`).
- **Consistência:** texto e vetores de um documento são gravados na mesma transação.

---

## Fragmentação de textos legais

Leis copiadas do Planalto ("texto compilado") trazem todas as redações que cada dispositivo já teve, em ordem cronológica, inclusive as de medidas provisórias que perderam a vigência. Quando o texto tem vários `Art. N` no início de linha, a ingestão (`backend/app/services/texto_legal.py`):

1. **Mantém só a redação vigente:** reconhece a hierarquia artigo > parágrafo > inciso > alínea > item; quando um dispositivo reaparece sob o mesmo pai, a redação nova substitui a anterior (com os filhos dela).
2. **Resume os revogados:** um dispositivo revogado vira só rótulo + nota, por exemplo `IV - (Revogada pela Lei nº 9.032, de 1995)`, e os filhos dele são descartados.
3. **Divide por artigo:** um chunk nunca mistura dois artigos. Um artigo longo é dividido nos limites dos dispositivos, e cada chunk repete o caput (resumido) no início.
4. **Entrega o artigo inteiro ao LLM:** a busca continua sobre os chunks pequenos, mas o artigo do chunk encontrado é remontado e enviado inteiro ao modelo, se tiver até `ARTIGO_MAX_CHARS` caracteres. Artigos repetidos entre os resultados contam uma vez só.

Títulos, capítulos e seções são descartados. Ementa e assinaturas são fragmentadas por parágrafos, assim como documentos que não são leis.

---

## Testes

```bash
cd backend
pytest tests/ -v
```

A suíte cobre fragmentação de texto (incluindo redação vigente e divisão por artigo das leis), leitura de arquivos, embeddings por API, indexação em segundo plano, integridade (hash anti-duplicação), token de acesso, serviço de recuperação/geração e rotas da API, usando mocks para o banco e os serviços externos.
