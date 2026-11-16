from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil, os, tempfile

from app.core.database import get_db
from app.models.models import STATUS_PROCESSANDO, Chunk, Documento
from app.services import rag_service, ingestao_service

router = APIRouter()


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class PerguntaRequest(BaseModel):
    pergunta: str
    # o modelo não é escolhido aqui: vem de LLM_PROVEDORES, com plano B
    id_documento: Optional[int] = None
    top_k: Optional[int] = None


class DocumentoSchema(BaseModel):
    id: int
    titulo: str
    status: str                                # 'processando', 'concluido' ou 'erro'
    erro: Optional[str] = None
    criado_em: Optional[datetime] = None
    total_chunks: int = 0


# ─────────────────────────────────────────────
# Consulta
# ─────────────────────────────────────────────
@router.post("/consultar", tags=["RAG"])
def consultar(req: PerguntaRequest, db: Session = Depends(get_db)):
    if not req.pergunta.strip():
        raise HTTPException(status_code=400, detail="Pergunta não pode estar vazia.")
    try:
        return rag_service.responder_pergunta(
            db=db,
            pergunta=req.pergunta,
            top_k=req.top_k,
            id_documento=req.id_documento,
        )
    except rag_service.ErroProvedor as e:
        raise HTTPException(status_code=502, detail=str(e))


# ─────────────────────────────────────────────
# Histórico / Ingestão / Listagens
# ─────────────────────────────────────────────
@router.get("/historico", tags=["RAG"])
def historico(limite: int = 20, db: Session = Depends(get_db)):
    return rag_service.listar_historico(db=db, limite=limite)


@router.get("/historico/{id_consulta}", tags=["RAG"])
def detalhe_consulta(id_consulta: int, db: Session = Depends(get_db)):
    """Resposta salva e fontes de uma consulta do histórico (sem chamar o LLM)."""
    consulta = rag_service.obter_consulta(db=db, id_consulta=id_consulta)
    if consulta is None:
        raise HTTPException(status_code=404, detail="Consulta não encontrada.")
    return consulta


@router.get("/sugestoes", tags=["RAG"])
def sugestoes(limite: int = 3, db: Session = Depends(get_db)):
    """Perguntas de exemplo montadas a partir dos documentos indexados."""
    return rag_service.gerar_sugestoes(db=db, limite=max(1, min(limite, 10)))


@router.post("/ingerir", status_code=202, tags=["Ingestão"])
def ingerir_documento(
    background_tasks: BackgroundTasks,
    arquivo: UploadFile = File(...),
    titulo: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Lê o arquivo e registra o documento na hora; chunks e embeddings são
    gerados em segundo plano. Acompanhe pelo status em GET /documentos.
    """
    extensoes_aceitas = {".pdf", ".txt", ".html", ".htm"}
    ext = os.path.splitext(arquivo.filename)[1].lower()
    if ext not in extensoes_aceitas:
        raise HTTPException(status_code=400, detail=f"Formato não suportado. Use: {extensoes_aceitas}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(arquivo.file, tmp)
        tmp_path = tmp.name

    try:
        doc, precisa_processar = ingestao_service.registrar_documento(
            db=db, caminho=tmp_path, titulo=titulo,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)

    if precisa_processar:
        background_tasks.add_task(ingestao_service.processar_documento, doc.id)
        mensagem = "Documento recebido. A indexação continua em segundo plano."
    else:
        mensagem = "Este arquivo já foi enviado antes."
    return {
        "mensagem": mensagem,
        "documento_id": doc.id,
        "titulo": doc.titulo,
        "status": doc.status,
    }


@router.get("/documentos", response_model=List[DocumentoSchema], tags=["Ingestão"])
def listar_documentos(db: Session = Depends(get_db)):
    docs = db.query(Documento).order_by(Documento.criado_em.desc()).all()
    contagem = dict(
        db.query(Chunk.id_documento, func.count(Chunk.id)).group_by(Chunk.id_documento).all()
    )
    return [
        DocumentoSchema(
            id=d.id, titulo=d.titulo, status=d.status, erro=d.erro,
            criado_em=d.criado_em, total_chunks=contagem.get(d.id, 0),
        )
        for d in docs
    ]


def _documento_editavel(db: Session, id_documento: int) -> Documento:
    doc = db.get(Documento, id_documento)
    if doc is None:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")
    if doc.status == STATUS_PROCESSANDO:
        raise HTTPException(status_code=409, detail="Aguarde o fim da indexação deste documento.")
    return doc


@router.delete("/documentos/{id_documento}", status_code=204, tags=["Ingestão"])
def excluir_documento(id_documento: int, db: Session = Depends(get_db)):
    """Remove o documento e seus chunks. O histórico de consultas é mantido."""
    ingestao_service.excluir_documento(db, _documento_editavel(db, id_documento))
    return Response(status_code=204)


@router.post("/documentos/{id_documento}/reindexar", status_code=202, tags=["Ingestão"])
def reindexar_documento(
    id_documento: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Refaz chunks e embeddings a partir do texto já salvo do documento."""
    doc = _documento_editavel(db, id_documento)
    ingestao_service.preparar_reindexacao(db, doc)
    background_tasks.add_task(ingestao_service.processar_documento, doc.id)
    return {"documento_id": doc.id, "titulo": doc.titulo, "status": doc.status}

