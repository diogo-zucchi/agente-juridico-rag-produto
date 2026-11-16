from pydantic_settings import BaseSettings
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent.parent.parent / ".env"

class Settings(BaseSettings):
    # 127.0.0.1 e não localhost: o compose só publica o banco em IPv4.
    # Porta 5433: a 5432 fica com o banco do TCC.
    database_url: str = "postgresql+psycopg://postgres:senha@127.0.0.1:5433/agente_juridico"

    # ── LLMs ──────────────────────────────────────────────
    maritaca_api_key: str = ""
    maritaca_model: str = "sabia-4"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # ── Embeddings (API compatível com a OpenAI) ──────────
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    # vazio: usa a OPENAI_API_KEY
    embedding_api_key: str = ""
    # dimensão dos vetores do modelo acima; deve bater com vector(N) em chunks.embedding
    embedding_dim: int = 1536

    # ── Acesso ────────────────────────────────────────────
    # se definido, as rotas /api/v1 exigem o cabeçalho X-API-Key com este valor
    api_token: str = ""
    # origens extras liberadas no CORS, separadas por vírgula; vazio: só a
    # própria API (o frontend é servido por ela)
    cors_origins: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    top_k_results: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    # artigo de lei recuperado vai inteiro ao LLM se tiver até este tamanho
    artigo_max_chars: int = 4000

    log_level: str = "INFO"
    # vazio: log só no stdout (o que plataformas de hospedagem coletam)
    log_file: str = ""

    class Config:
        env_file = str(ENV_FILE)
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()