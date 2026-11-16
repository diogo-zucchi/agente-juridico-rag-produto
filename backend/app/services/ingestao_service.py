"""
Serviço de Ingestão de Documentos Jurídicos.

Responsável por:
  1. Ler arquivos (PDF, TXT, HTML)
  2. Dividir em chunks (em leis: só a redação vigente, por artigo)
  3. Gerar embeddings
  4. Salvar texto e vetores no PostgreSQL (pgvector)
"""
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from openai import OpenAI
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.models import (
    STATUS_CONCLUIDO, STATUS_ERRO, STATUS_PROCESSANDO, Chunk, Consulta, Documento,
    RespostaDocumento,
)
from app.services.texto_legal import (
    chave_do_artigo, dividir_artigo, eh_texto_legal, extrair_artigos, juntar_artigo,
    rotulo_do_artigo,
)

settings = get_settings()


# ─────────────────────────────────────────────
# Embeddings por API (compatível com a OpenAI)
# ─────────────────────────────────────────────
# textos por requisição: bem abaixo dos limites de itens e tokens da API
_LOTE_EMBEDDINGS = 128
_cliente_embeddings: OpenAI | None = None


def _get_cliente_embeddings() -> OpenAI:
    global _cliente_embeddings
    if _cliente_embeddings is None:
        _cliente_embeddings = OpenAI(
            api_key=settings.embedding_api_key or settings.openai_api_key,
            base_url=settings.embedding_base_url,
        )
    return _cliente_embeddings


def gerar_embeddings(textos: List[str]) -> np.ndarray:
    """Embeddings normalizados (norma 1): produto interno = cosseno."""
    if not textos:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)
    cliente = _get_cliente_embeddings()
    vetores = []
    for i in range(0, len(textos), _LOTE_EMBEDDINGS):
        resposta = cliente.embeddings.create(
            model=settings.embedding_model,
            input=textos[i : i + _LOTE_EMBEDDINGS],
        )
        vetores.extend(item.embedding for item in resposta.data)

    matriz = np.asarray(vetores, dtype=np.float32)
    if matriz.shape[1] != settings.embedding_dim:
        raise ValueError(
            f"O modelo de embeddings gera vetores de dimensão {matriz.shape[1]}, mas a "
            f"coluna chunks.embedding espera {settings.embedding_dim}. "
            "Ajuste EMBEDDING_DIM e recrie a tabela chunks."
        )
    return matriz / np.linalg.norm(matriz, axis=1, keepdims=True)


# ─────────────────────────────────────────────
# Leitura de arquivos
# ─────────────────────────────────────────────

def _ler_pdf(caminho: str) -> str:
    texto = ""

    try:
        import fitz  # PyMuPDF
        with fitz.open(caminho) as doc:
            texto = "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning(f"PyMuPDF não disponível/falhou ({e}); tentando pypdf...")

    if not texto.strip():
        from pypdf import PdfReader
        reader = PdfReader(caminho)
        texto = "\n".join(p.extract_text() or "" for p in reader.pages)

    if not texto.strip():
        raise ValueError(
            "Não foi possível extrair texto deste PDF. Ele pode ser digitalizado "
            "(imagem), o que exigiria OCR. Envie o documento em .txt ou .html."
        )
    return texto


def _ler_txt(caminho: str) -> str:
    return Path(caminho).read_text(encoding="utf-8", errors="ignore")


def _ler_html(caminho: str) -> str:
    from bs4 import BeautifulSoup
    html = Path(caminho).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")


def ler_arquivo(caminho: str) -> str:
    ext = Path(caminho).suffix.lower()
    if ext == ".pdf":
        return _ler_pdf(caminho)
    elif ext in (".txt", ".md"):
        return _ler_txt(caminho)
    elif ext in (".html", ".htm"):
        return _ler_html(caminho)
    else:
        raise ValueError(f"Formato não suportado: {ext}")


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def _dividir_em_chunks(texto: str) -> List[str]:
    """
    Textos legais (com 'Art. N') são limpos das redações revogadas e
    divididos por artigo; os demais, por parágrafos.
    """
    if not eh_texto_legal(texto):
        return _dividir_por_paragrafos(texto)

    artigos, texto_fora = extrair_artigos(texto)
    chunks = _dividir_por_paragrafos(texto_fora) if texto_fora else []
    for linhas in artigos:
        chunks.extend(dividir_artigo(linhas, settings.chunk_size, settings.chunk_overlap))
    return chunks


def _dividir_por_paragrafos(texto: str) -> List[str]:
    """
    Divide o texto em chunks com sobreposição.
    Tenta respeitar parágrafos antes de cortar por caracteres.
    """
    tamanho  = settings.chunk_size
    overlap  = settings.chunk_overlap
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]

    chunks  = []
    buffer  = ""

    for paragrafo in paragrafos:
        if len(buffer) + len(paragrafo) <= tamanho:
            buffer += (" " if buffer else "") + paragrafo
        else:
            if buffer:
                chunks.append(buffer)
            # Parágrafo maior que o chunk? Divide por caractere.
            if len(paragrafo) > tamanho:
                for i in range(0, len(paragrafo), tamanho - overlap):
                    chunks.append(paragrafo[i : i + tamanho])
                # o buffer já foi emitido: zera para não duplicá-lo
                # no próximo chunk
                buffer = ""
            else:
                buffer = paragrafo

    if buffer:
        chunks.append(buffer)

    return chunks


# ─────────────────────────────────────────────
# Pipeline principal de ingestão
# ─────────────────────────────────────────────

#   registrar_documento   rápido: lê o arquivo, confere o hash e grava o
#                         documento com status "processando"
#   processar_documento   demorado: chunks + embeddings; roda em segundo
#                         plano na API, para o upload não estourar o timeout

def registrar_documento(
    db: Session,
    caminho: str,
    titulo: str,
    id_fonte: int | None = None,
) -> Tuple[Documento, bool]:
    """
    Devolve (documento, precisa_processar). Um arquivo já enviado é
    reaproveitado pelo hash; se a indexação dele tinha falhado, é refeita.
    """
    logger.info(f"Lendo arquivo: {caminho}")
    texto = ler_arquivo(caminho)

    hash_doc = hashlib.sha256(texto.encode()).hexdigest()
    existente = db.query(Documento).filter_by(hash_integridade=hash_doc).first()
    if existente:
        if existente.status != STATUS_ERRO:
            logger.warning(f"Documento já enviado (hash={hash_doc[:8]}…). Ignorando.")
            return existente, False
        existente.status, existente.erro = STATUS_PROCESSANDO, None
        db.commit()
        return existente, True

    doc = Documento(
        titulo=titulo,
        texto=texto,
        id_fonte=id_fonte,
        hash_integridade=hash_doc,
        arquivo_origem=os.path.basename(caminho),
        status=STATUS_PROCESSANDO,
    )
    db.add(doc)
    db.commit()
    return doc, True


def processar_documento(id_documento: int) -> None:
    """
    Gera chunks e embeddings do documento. Abre a própria sessão, porque
    roda depois que a requisição terminou. Chunks e vetores são gravados
    na mesma transação: ou o documento inteiro fica indexado, ou nada.
    """
    db = SessionLocal()
    try:
        doc = db.get(Documento, id_documento)
        lista_chunks = _dividir_em_chunks(doc.texto)
        logger.info(f"'{doc.titulo}' dividido em {len(lista_chunks)} chunks.")

        embeddings = gerar_embeddings(lista_chunks)
        for idx, (chunk_texto, emb) in enumerate(zip(lista_chunks, embeddings)):
            db.add(Chunk(
                id_documento=doc.id,
                texto_chunk=chunk_texto,
                indice_chunk=idx,
                embedding=emb,
            ))
        doc.status = STATUS_CONCLUIDO
        db.commit()
        logger.info(f"Documento '{doc.titulo}' indexado com sucesso ({len(lista_chunks)} chunks).")
    except Exception as e:
        logger.exception(f"Falha ao indexar o documento {id_documento}: {e}")
        db.rollback()
        doc = db.get(Documento, id_documento)
        if doc is not None:
            doc.status, doc.erro = STATUS_ERRO, str(e)[:1000]
            db.commit()
    finally:
        db.close()


def _apagar_chunks(db: Session, id_documento: int) -> None:
    """Apaga os chunks do documento, soltando antes as respostas antigas que
    os citavam (elas continuam no histórico, só sem o vínculo)."""
    ids_chunks = select(Chunk.id).where(Chunk.id_documento == id_documento).scalar_subquery()
    db.execute(
        update(RespostaDocumento)
        .where(RespostaDocumento.id_chunk.in_(ids_chunks))
        .values(id_chunk=None)
    )
    db.execute(delete(Chunk).where(Chunk.id_documento == id_documento))


def excluir_documento(db: Session, doc: Documento) -> None:
    """Remove o documento e seus chunks; consultas feitas sobre ele ficam no histórico."""
    _apagar_chunks(db, doc.id)
    db.execute(
        update(Consulta).where(Consulta.id_documento == doc.id).values(id_documento=None)
    )
    db.execute(delete(Documento).where(Documento.id == doc.id))
    db.commit()
    logger.info(f"Documento {doc.id} excluído.")


def preparar_reindexacao(db: Session, doc: Documento) -> None:
    """Descarta os chunks e volta o documento a 'processando'; a indexação
    em si é feita depois por processar_documento, a partir do texto salvo."""
    _apagar_chunks(db, doc.id)
    doc.status, doc.erro = STATUS_PROCESSANDO, None
    db.commit()


def ingerir_documento(
    db: Session,
    caminho: str,
    titulo: str,
    id_fonte: int | None = None,
) -> Documento:
    """Pipeline completo, de uma vez (usado pelo script de indexação em lote)."""
    doc, precisa_processar = registrar_documento(
        db, caminho, titulo, id_fonte=id_fonte,
    )
    if precisa_processar:
        processar_documento(doc.id)
        db.refresh(doc)
        if doc.status == STATUS_ERRO:
            raise RuntimeError(doc.erro)
    return doc


# ─────────────────────────────────────────────
# Busca semântica (usada pelo retriever)
# ─────────────────────────────────────────────

def buscar_chunks(
    db: Session,
    pergunta: str,
    top_k: int | None = None,
    id_documento: int | None = None,
) -> List[Tuple[Chunk, float]]:
    """
    Retorna lista de (chunk, score) ordenada por similaridade.
    Se id_documento for informado, a busca é restrita àquele documento.

    Busca exata (sem índice aproximado): o score é o produto interno
    entre embeddings normalizados, o mesmo do antigo IndexFlatIP do FAISS.
    """
    k = top_k or settings.top_k_results
    vetor_pergunta = gerar_embeddings([pergunta])[0]

    # operador <#> do pgvector: devolve o produto interno NEGATIVO
    distancia = Chunk.embedding.max_inner_product(vetor_pergunta)

    stmt = (
        select(Chunk, distancia)
        .where(Chunk.embedding.isnot(None))
        .order_by(distancia)
        .limit(k)
    )
    if id_documento is not None:
        stmt = stmt.where(Chunk.id_documento == id_documento)

    resultados = db.execute(stmt).all()
    if not resultados:
        logger.warning("Nenhum chunk indexado encontrado. Indexe documentos primeiro.")

    return [(chunk, -float(dist)) for chunk, dist in resultados]


def texto_do_artigo(db: Session, id_documento: int, rotulo: str) -> str:
    """
    Remonta o artigo inteiro a partir dos chunks do documento que começam
    com o rótulo (todo chunk de um artigo começa com ele).
    """
    stmt = (
        select(Chunk.texto_chunk)
        .where(Chunk.id_documento == id_documento)
        .where(Chunk.texto_chunk.startswith(rotulo, autoescape=True))
        .order_by(Chunk.indice_chunk)
    )
    # startswith("Art. 1") também pega "Art. 10": confere o rótulo exato
    textos = [t for t in db.scalars(stmt).all() if rotulo_do_artigo(t) == rotulo]
    return juntar_artigo(textos)


def buscar_artigos(
    db: Session,
    chaves: List[str],
    ids_documentos: List[int] | None = None,
) -> List[Tuple[str, Chunk]]:
    """
    Busca exata de artigos pelo número (chaves no formato de chave_do_artigo,
    como '21' ou '6-B'). Devolve (chave, primeiro chunk do artigo) para cada
    documento que tem o artigo, em ordem de documento. Sem ids_documentos,
    procura em todos.

    O banco filtra pelo número ("Art. 21" seguido de algo que não seja
    dígito, para não pegar o Art. 210); a letra e o ordinal são conferidos
    aqui, com a mesma regra usada na ingestão.
    """
    if not chaves or ids_documentos == []:
        return []
    numeros = sorted({c.split("-")[0] for c in chaves}, key=int)
    padrao = rf"^Art\.?\s*({'|'.join(numeros)})(\D|$)"
    stmt = (
        select(Chunk)
        .where(Chunk.texto_chunk.regexp_match(padrao))
        .order_by(Chunk.id_documento, Chunk.indice_chunk)
    )
    if ids_documentos is not None:
        stmt = stmt.where(Chunk.id_documento.in_(ids_documentos))

    primeiros: dict = {}
    for chunk in db.scalars(stmt).all():
        chave = chave_do_artigo(chunk.texto_chunk)
        if chave in chaves:
            primeiros.setdefault((chunk.id_documento, chave), chunk)
    return [(chave, chunk) for (_, chave), chunk in primeiros.items()]
