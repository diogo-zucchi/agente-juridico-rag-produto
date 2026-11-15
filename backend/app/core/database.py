from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import get_settings
from loguru import logger

settings = get_settings()

url = settings.database_url
# Neon, Supabase e afins entregam postgres:// ou postgresql:// sem o driver
if url.startswith("postgres://"):
    url = url.replace("postgres://", "postgresql+psycopg://", 1)
elif url.startswith("postgresql://"):
    url = url.replace("postgresql://", "postgresql+psycopg://", 1)
elif not "postgresql+psycopg" in url:
    url = url.replace("postgresql+psycopg2", "postgresql+psycopg", 1)

engine = create_engine(
    url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    # banco inacessível: falha em segundos em vez de travar a inicialização
    connect_args={"connect_timeout": 10},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


FONTES_INICIAIS = [
    ("Superior Tribunal de Justiça", "STJ", "https://www.stj.jus.br"),
    ("Supremo Tribunal Federal", "STF", "https://portal.stf.jus.br"),
    ("Planalto - Legislação Federal", "legislação", "https://www.planalto.gov.br"),
    ("Tribunal Superior do Trabalho", "TST", "https://www.tst.jus.br"),
]


def criar_tabelas():
    """
    Habilita o pgvector, cria as tabelas que ainda não existem e cadastra
    as fontes iniciais num banco vazio. Não altera tabelas já existentes:
    mudou o esquema, recrie o banco.
    """
    from app.models import models  # noqa: F401 — registra os modelos no Base

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        if db.query(models.Fonte).count() == 0:
            db.add_all(models.Fonte(nome=n, tipo=t, url_base=u) for n, t, u in FONTES_INICIAIS)
            db.commit()


def marcar_indexacoes_interrompidas():
    """
    A indexação roda em segundo plano, no próprio processo da API: se o
    servidor reiniciou no meio dela, o documento ficou 'processando' para
    sempre. Pressupõe uma única instância da API.
    """
    from app.models.models import Documento, STATUS_ERRO, STATUS_PROCESSANDO

    with SessionLocal() as db:
        n = (db.query(Documento)
               .filter_by(status=STATUS_PROCESSANDO)
               .update({"status": STATUS_ERRO,
                        "erro": "Indexação interrompida pelo reinício do servidor. Envie o arquivo de novo."}))
        db.commit()
    if n:
        logger.warning(f"{n} indexação(ões) interrompida(s) marcada(s) como erro.")


def verificar_conexao():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Conexão com PostgreSQL estabelecida.")
        return True
    except Exception as e:
        logger.error(f"❌ Falha ao conectar ao PostgreSQL: {e}")
        return False