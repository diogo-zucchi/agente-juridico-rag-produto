# Agente Jurídico

Consulta jurídica em linguagem natural, com respostas fundamentadas nos documentos que você indexa.

Você envia leis, pareceres ou decisões e faz perguntas como faria a um colega. O Agente Jurídico encontra os trechos relevantes, escreve uma resposta baseada apenas neles e cita cada fonte, para que qualquer informação possa ser conferida no texto original.

---

## Funcionalidades

- **Perguntas em linguagem natural:** a busca é semântica, então a pergunta não precisa repetir as palavras exatas da lei.
- **Respostas com fontes:** cada resposta indica os trechos usados (`[1]`, `[2]`…), com o documento de onde vieram e o grau de relevância.
- **Artigos citados pelo número:** perguntas como "o que diz o Art. 20 da Lei 8.742?" trazem o texto exato do artigo.
- **Leis tratadas como leis:** em textos compilados do Planalto, só a redação vigente é indexada; dispositivos revogados viram uma nota curta.
- **Filtro por documento:** a busca pode ser restrita a um único documento quando a pergunta exige precisão.
- **Sugestões de perguntas:** geradas a partir dos documentos indexados, para quem está começando.
- **Histórico:** consultas anteriores ficam salvas e podem ser reabertas sem gerar a resposta de novo.
- **Gestão de documentos:** envio de PDF, TXT ou HTML, com indexação em segundo plano, reindexação e exclusão.
- **Modelo reserva:** se o modelo principal falhar, a pergunta é respondida automaticamente por outro.

---

## Como funciona

1. **Indexação:** o documento enviado é dividido em trechos, e cada trecho vira um vetor (embedding) guardado no banco junto do texto.
2. **Busca:** a pergunta também vira um vetor, e o banco devolve os trechos mais próximos. Artigos citados pelo número entram primeiro, com o texto exato.
3. **Resposta:** os trechos encontrados vão para o modelo de linguagem com a instrução de responder apenas com base neles e citar as fontes.

---

## Modelos de linguagem

| Provedor | Modelo | Papel |
|----------|--------|-------|
| Maritaca | `sabia-4` | **Principal**: modelo brasileiro, com pré-treino em português e legislação. |
| OpenAI | `gpt-4o-mini` | **Reserva**: responde quando o principal falha. |

O servidor decide qual modelo usar. `LLM_PROVEDORES` define a ordem de tentativa (padrão: `maritaca,openai`). Se o primeiro falhar (fora do ar, chave inválida, limite de uso ou demora acima de `LLM_TIMEOUT` segundos), a mesma pergunta vai para o próximo. A resposta informa qual modelo respondeu.

Os dois usam a API compatível com a OpenAI, então trocar de modelo é questão de configuração: URL base, chave e nome do modelo.

---

## Arquitetura

- **Interface:** página única em HTML, CSS e JavaScript (`frontend/index.html`), servida pela própria API em `/`.
- **API:** Python 3.13 e FastAPI, em camadas (rotas → serviços → persistência).
- **Banco:** PostgreSQL com a extensão pgvector. Dados e vetores ficam no mesmo banco.
- **Embeddings:** `text-embedding-3-small` da OpenAI (1536 dimensões). Qualquer provedor com API compatível com a OpenAI serve (ver `EMBEDDING_*` na configuração).

---

## Começando

### Pré-requisitos

- Python 3.13
- Um PostgreSQL com pgvector, como o [Neon](https://neon.tech) ou o [Supabase](https://supabase.com)
- Chaves de API da Maritaca e da OpenAI

### Instalação

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows (PowerShell)
# source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

### Configuração

Crie um arquivo `.env` na raiz do projeto. O mínimo é isto:

```env
DATABASE_URL=postgresql://usuario:senha@host/banco?sslmode=require
MARITACA_API_KEY=sua_chave_maritaca
OPENAI_API_KEY=sua_chave_openai
```

A URL do banco pode ser colada como o provedor a entrega (`postgresql://...`): a API ajusta o driver sozinha.

<details>
<summary>Todas as opções</summary>

```env
# ── Banco de dados ───────────────────────────────────────
DATABASE_URL=postgresql://usuario:senha@host/banco?sslmode=require

# ── Modelos de linguagem ─────────────────────────────────
LLM_PROVEDORES=maritaca,openai   # ordem de tentativa
LLM_TIMEOUT=60                   # segundos antes de passar ao próximo

MARITACA_API_KEY=sua_chave_maritaca
MARITACA_MODEL=sabia-4

OPENAI_API_KEY=sua_chave_openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# ── Embeddings (API compatível com a OpenAI) ─────────────
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=          # vazio: usa a OPENAI_API_KEY
EMBEDDING_DIM=1536

# ── Acesso ───────────────────────────────────────────────
API_TOKEN=                  # vazio: API aberta, sem token
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

`EMBEDDING_MODEL` e `EMBEDDING_DIM` andam juntos e definem a coluna `vector(N)` do banco. Para trocar de modelo de embeddings, recrie o banco e indexe os documentos de novo.

</details>

### Execução

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8001
```

Ao iniciar, a API habilita o pgvector e cria as tabelas que faltarem.

- Interface: `http://localhost:8001/`
- Documentação interativa da API: `http://localhost:8001/docs`

Com `--reload`, a API reinicia a cada alteração no código. A página é lida a cada acesso, então basta recarregar o navegador.

---

## Uso

1. **Indexe documentos:** na área "Indexar documento", envie um PDF, TXT ou HTML com um título. A indexação continua em segundo plano, e o andamento aparece na lista de documentos.
2. **Pergunte:** digite a pergunta e clique em **Consultar** (ou `Ctrl + Enter`). A resposta vem com as fontes numeradas.
3. **Refine, se precisar:** "Filtrar por documento" restringe a busca a um documento.

A pasta `leis/` traz três leis prontas para começar: LC 142/2013, Lei 8.213/91 e Lei 8.742/93. Basta enviá-las pela página.

---

## Publicação

O projeto roda em duas peças:

1. **Banco:** um PostgreSQL gerenciado com pgvector, como [Neon](https://neon.tech) ou [Supabase](https://supabase.com). As tabelas são criadas na primeira inicialização.
2. **API:** uma plataforma que rode Python a partir do repositório, como [Render](https://render.com) ou [Railway](https://railway.app).

### Neon + Render

1. No Neon, crie um projeto na região **AWS US East (N. Virginia)** e copie a connection string.
2. No Render, use **New → Blueprint** e escolha este repositório. O `render.yaml` cria o serviço na mesma região do banco.
3. Informe `DATABASE_URL`, `MARITACA_API_KEY` e `OPENAI_API_KEY`. O `API_TOKEN` é gerado pelo Render; copie-o em **Environment** para acessar a página.
4. Quando o deploy terminar, `https://<serviço>.onrender.com/health` deve responder `"database": true`.

A cada push no `main`, o Render publica a versão nova automaticamente.

No plano gratuito, o serviço dorme após 15 minutos sem acesso e leva cerca de um minuto para acordar.

### Outras plataformas

- **Build:** `pip install -r requirements.txt`
- **Start:** `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Python:** 3.13 (arquivo `.python-version`)
- **Health check:** `/health`
- **Variáveis:** `DATABASE_URL`, `MARITACA_API_KEY`, `OPENAI_API_KEY` e `API_TOKEN` (um valor longo e aleatório)

### Acesso

Com `API_TOKEN` definido, as rotas `/api/v1` exigem o cabeçalho `X-API-Key`. A página pede o token na primeira chamada recusada e o guarda no navegador. A página e o `/health` continuam públicos.

### Uma instância

A indexação roda em segundo plano no próprio processo da API: o envio responde na hora, e o status do documento (`processando`, `concluido` ou `erro`) aparece na lista. Se o servidor reiniciar durante uma indexação, o documento fica com erro e basta reindexá-lo pela página. Para rodar mais de uma instância, a indexação precisaria de uma fila.

---

## API

Prefixo: `/api/v1`

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/consultar` | Responde a uma pergunta, com fontes. Responde `502` se todos os modelos ou a API de embeddings falharem. |
| `GET` | `/historico` | Consultas recentes. |
| `GET` | `/historico/{id}` | Resposta salva e fontes de uma consulta, sem chamar o modelo de novo. |
| `GET` | `/sugestoes` | Perguntas sugeridas a partir dos documentos indexados. |
| `POST` | `/ingerir` | Recebe um documento e o indexa em segundo plano (responde `202`). |
| `GET` | `/documentos` | Documentos, status da indexação e total de trechos de cada um. |
| `POST` | `/documentos/{id}/reindexar` | Refaz trechos e embeddings a partir do texto salvo (responde `202`). |
| `DELETE` | `/documentos/{id}` | Exclui o documento e seus trechos; o histórico de consultas é mantido. |

---

## Detalhes técnicos

### Vetores

Os vetores ficam na coluna `chunks.embedding` (`vector(1536)`), ao lado do texto de cada trecho.

- **Busca:** produto interno (operador `<#>`) sobre embeddings normalizados, o que equivale à similaridade de cosseno.
- **Busca exata:** não há índice aproximado. Com poucos milhares de trechos a varredura é rápida e o resultado é determinístico. Se o volume crescer, crie um índice HNSW: `CREATE INDEX ON chunks USING hnsw (embedding vector_ip_ops);`
- **Filtro por documento:** aplicado na própria consulta SQL.
- **Consistência:** texto e vetores de um documento são gravados na mesma transação.

### Textos legais

Leis copiadas do Planalto ("texto compilado") trazem todas as redações que cada dispositivo já teve, inclusive as de medidas provisórias que perderam a vigência. Quando o texto tem vários `Art. N` no início de linha, a indexação (`backend/app/services/texto_legal.py`):

1. **Mantém só a redação vigente:** reconhece a hierarquia artigo > parágrafo > inciso > alínea > item. Quando um dispositivo reaparece sob o mesmo pai, a redação nova substitui a anterior.
2. **Resume os revogados:** um dispositivo revogado vira só rótulo e nota, por exemplo `IV - (Revogada pela Lei nº 9.032, de 1995)`.
3. **Divide por artigo:** um trecho nunca mistura dois artigos. Um artigo longo é dividido nos limites dos dispositivos, e cada trecho repete o caput resumido no início.
4. **Entrega o artigo inteiro ao modelo:** a busca usa os trechos pequenos, mas o artigo encontrado é remontado e enviado inteiro, se tiver até `ARTIGO_MAX_CHARS` caracteres.

Títulos, capítulos e seções são descartados. Ementa, assinaturas e documentos que não são leis são divididos por parágrafos.

---

## Testes

```bash
cd backend
pytest tests/ -v
```

A suíte cobre a divisão de textos (incluindo redação vigente e divisão por artigo), leitura de arquivos, embeddings, indexação em segundo plano, controle de duplicados, token de acesso, busca, geração com modelo reserva e as rotas da API. Banco e serviços externos são simulados, então os testes não precisam de chaves nem de conexão.
