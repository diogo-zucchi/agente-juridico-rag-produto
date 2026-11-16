"""
Modelos ORM — são a definição do esquema: criar_tabelas() cria as
tabelas a partir deles, tanto no Docker local quanto num banco hospedado.
"""
from sqlalchemy import (
    Column, Integer, String, Text, Date, Float,
    DateTime, ForeignKey, func
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.config import get_settings
from app.core.database import Base


class Fonte(Base):
    __tablename__ = "fontes"

    id        = Column(Integer, primary_key=True, index=True)
    nome      = Column(String(255), nullable=False)
    tipo      = Column(String(100), nullable=False)
    url_base  = Column(Text)
    criado_em = Column(DateTime, default=func.now())

    documentos = relationship("Documento", back_populates="fonte")


# status da indexação de um documento
STATUS_PROCESSANDO = "processando"
STATUS_CONCLUIDO   = "concluido"
STATUS_ERRO        = "erro"


class Documento(Base):
    __tablename__ = "documentos"

    id               = Column(Integer, primary_key=True, index=True)
    titulo           = Column(String(500), nullable=False)
    texto            = Column(Text, nullable=False)
    data_publicacao  = Column(Date)
    id_fonte         = Column(Integer, ForeignKey("fontes.id", ondelete="SET NULL"))
    hash_integridade = Column(String(64), unique=True)
    arquivo_origem   = Column(String(500))
    status           = Column(String(20), nullable=False, default=STATUS_CONCLUIDO)
    # mensagem da falha, quando status = 'erro'
    erro             = Column(Text, nullable=True)
    criado_em        = Column(DateTime, default=func.now())

    fonte  = relationship("Fonte", back_populates="documentos")
    chunks = relationship("Chunk", back_populates="documento", cascade="all, delete")


class Chunk(Base):
    __tablename__ = "chunks"

    id           = Column(Integer, primary_key=True, index=True)
    # indexado: filtro da busca restrita a um documento
    id_documento = Column(Integer, ForeignKey("documentos.id", ondelete="CASCADE"), index=True)
    texto_chunk  = Column(Text, nullable=False)
    indice_chunk = Column(Integer, nullable=False)
    # vetor normalizado do texto_chunk (pgvector). Sem índice vetorial de
    # propósito: com poucos milhares de chunks a busca exata é rápida e
    # determinística. Se o volume crescer, crie um HNSW:
    #   CREATE INDEX ON chunks USING hnsw (embedding vector_ip_ops);
    embedding    = Column(Vector(get_settings().embedding_dim))
    criado_em    = Column(DateTime, default=func.now())

    documento = relationship("Documento", back_populates="chunks")
    respostas = relationship("RespostaDocumento", back_populates="chunk")


class Consulta(Base):
    __tablename__ = "consultas"

    id          = Column(Integer, primary_key=True, index=True)
    pergunta    = Column(Text, nullable=False)
    data_hora   = Column(DateTime, default=func.now())
    # use_alter: consultas e respostas se referenciam (FK criada por ALTER TABLE)
    id_resposta = Column(Integer, ForeignKey("respostas.id", ondelete="SET NULL", use_alter=True),
                         nullable=True)
    # documento consultado (quando a busca é restrita a 1 documento)
    id_documento = Column(Integer, ForeignKey("documentos.id", ondelete="SET NULL"), nullable=True)

    resposta = relationship("Resposta", back_populates="consulta",
                            foreign_keys=[id_resposta])
    documento = relationship("Documento", foreign_keys=[id_documento])


class Resposta(Base):
    __tablename__ = "respostas"

    id             = Column(Integer, primary_key=True, index=True)
    texto_resposta = Column(Text, nullable=False)
    # qual LLM gerou esta resposta: 'maritaca' ou 'openai'
    modelo_llm     = Column(String(50), nullable=True)
    data_hora      = Column(DateTime, default=func.now())
    id_consulta    = Column(Integer, ForeignKey("consultas.id", ondelete="CASCADE"))

    consulta  = relationship("Consulta", back_populates="resposta",
                              foreign_keys=[Consulta.id_resposta])
    documentos_utilizados = relationship("RespostaDocumento", back_populates="resposta")


class RespostaDocumento(Base):
    __tablename__ = "resposta_documento"

    id          = Column(Integer, primary_key=True, index=True)
    id_resposta = Column(Integer, ForeignKey("respostas.id", ondelete="CASCADE"))
    id_chunk    = Column(Integer, ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    score       = Column(Float)
    criado_em   = Column(DateTime, default=func.now())

    resposta = relationship("Resposta", back_populates="documentos_utilizados")
    chunk    = relationship("Chunk", back_populates="respostas")

