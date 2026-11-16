from typing import List, Dict, Any, Callable
from openai import OpenAI
from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.models import Consulta, Resposta, RespostaDocumento
from app.services.ingestao_service import buscar_chunks, texto_do_artigo
from app.services.texto_legal import rotulo_do_artigo

PROMPT_JURIDICO = """Você é um assistente jurídico especializado no direito brasileiro.
Utilize EXCLUSIVAMENTE os trechos de documentos jurídicos fornecidos abaixo.
Nunca invente informações. Se a resposta não estiver nos documentos, diga claramente.
Sempre indique as fontes utilizadas na sua resposta.

=== DOCUMENTOS JURÍDICOS RECUPERADOS ===
{contexto}
=========================================

Pergunta: {pergunta}

Resposta fundamentada:"""

# Provedores de LLM suportados
PROVEDORES = ("maritaca", "openai")

# Cache de invocadores por provedor (criados sob demanda)
_llms: Dict[str, Callable[[str], str]] = {}


# ─────────────────────────────────────────────
# Fábrica de LLMs (Maritaca e OpenAI)
# Ambas usam a interface compatível com a OpenAI,
# mudando apenas base_url, api_key e nome do modelo.
# ─────────────────────────────────────────────
def _criar_llm(provedor: str) -> Callable[[str], str]:
    settings = get_settings()

    if provedor == "maritaca":
        logger.info(f"Carregando LLM Maritaca ({settings.maritaca_model})")
        client = OpenAI(
            api_key=settings.maritaca_api_key,
            base_url="https://chat.maritaca.ai/api",
        )
        modelo = settings.maritaca_model

    elif provedor == "openai":
        logger.info(f"Carregando LLM OpenAI ({settings.openai_model})")
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        modelo = settings.openai_model

    else:
        raise ValueError(f"Provedor de LLM desconhecido: {provedor}")

    def invocar(prompt: str) -> str:
        response = client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    return invocar


def _get_llm(provedor: str) -> Callable[[str], str]:
    if provedor not in _llms:
        _llms[provedor] = _criar_llm(provedor)
    return _llms[provedor]


def _montar_contexto(chunks_com_score: List[Dict]) -> str:
    partes = []
    for i, item in enumerate(chunks_com_score, 1):
        fonte = item.get("fonte", "Fonte desconhecida")
        titulo = item.get("titulo", "Documento")
        texto = item.get("texto_chunk", "")
        partes.append(f"[Documento {i}] {titulo} ({fonte})\n{texto}")
    return "\n\n---\n\n".join(partes)


# ─────────────────────────────────────────────
# Recuperação de chunks (com opção de restringir
# a busca a um único documento)
# ─────────────────────────────────────────────
def _recuperar_chunks(
    db: Session,
    pergunta: str,
    k: int,
    id_documento: int | None = None,
) -> List[Dict]:
    """
    Retorna os chunks mais relevantes já enriquecidos com título/fonte.
    Se id_documento for informado, a busca é restrita àquele documento
    (filtro aplicado no próprio banco, junto da busca vetorial).

    Chunks de lei são buscados pequenos (embedding mais preciso), mas o LLM
    recebe o artigo inteiro: assim, achar o § 5º do Art. 16 traz também os
    incisos do caput. Artigos repetidos contam uma vez só (com o maior score).
    """
    max_chars = get_settings().artigo_max_chars
    # busca mais candidatos, porque vários podem ser do mesmo artigo
    resultados = buscar_chunks(db, pergunta, top_k=k * 3, id_documento=id_documento)

    chunks_com_score = []
    vistos = set()
    for chunk, score in resultados:
        texto, chave = chunk.texto_chunk, ("chunk", chunk.id)
        rotulo = rotulo_do_artigo(chunk.texto_chunk)
        if rotulo:
            artigo = texto_do_artigo(db, chunk.id_documento, rotulo)
            if artigo and len(artigo) <= max_chars:
                texto, chave = artigo, ("artigo", chunk.id_documento, rotulo)
        if chave in vistos:
            continue
        vistos.add(chave)

        doc = chunk.documento
        fonte = doc.fonte.nome if doc and doc.fonte else "Fonte desconhecida"
        chunks_com_score.append({
            "chunk_id": chunk.id,
            "titulo": doc.titulo if doc else "Documento",
            "fonte": fonte,
            "texto_chunk": texto,
            "score": score,
        })
        if len(chunks_com_score) == k:
            break
    return chunks_com_score


def _gerar_texto(provedor: str, chunks_com_score: List[Dict], pergunta: str) -> str:
    if not chunks_com_score:
        return (
            "Não encontrei documentos jurídicos relevantes para sua pergunta. "
            "Por favor, indexe documentos relacionados ao tema antes de consultar."
        )
    contexto = _montar_contexto(chunks_com_score)
    prompt = PROMPT_JURIDICO.format(contexto=contexto, pergunta=pergunta)
    logger.info(f"Gerando resposta com LLM: {provedor}")
    try:
        return _get_llm(provedor)(prompt)
    except Exception as e:
        logger.error(f"Erro ao chamar LLM {provedor}: {e}")
        return f"Ocorreu um erro ao gerar a resposta com {provedor}. Verifique a chave de API e o modelo configurado."


def _salvar_resposta(
    db: Session,
    consulta: Consulta,
    texto: str,
    provedor: str,
    fontes: List[Dict],
) -> Resposta:
    resposta = Resposta(
        texto_resposta=texto,
        modelo_llm=provedor,
        id_consulta=consulta.id,
    )
    db.add(resposta)
    db.flush()
    for item in fontes:
        db.add(RespostaDocumento(
            id_resposta=resposta.id,
            id_chunk=item["chunk_id"],
            score=item["score"],
        ))
    return resposta


# ─────────────────────────────────────────────
# Consulta
# ─────────────────────────────────────────────
def responder_pergunta(
    db: Session,
    pergunta: str,
    provedor: str = "maritaca",
    top_k: int | None = None,
    id_documento: int | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    k = top_k or settings.top_k_results
    if provedor not in PROVEDORES:
        provedor = "maritaca"

    fontes = _recuperar_chunks(db, pergunta, k, id_documento)
    texto = _gerar_texto(provedor, fontes, pergunta)

    consulta = Consulta(pergunta=pergunta, id_documento=id_documento)
    db.add(consulta)
    db.flush()

    resposta = _salvar_resposta(db, consulta, texto, provedor, fontes)
    consulta.id_resposta = resposta.id
    db.commit()

    logger.info(f"Resposta ({provedor}) salva (consulta_id={consulta.id}).")
    return {
        "consulta_id": consulta.id,
        "resposta_id": resposta.id,
        "modelo_llm": provedor,
        "resposta": texto,
        "fontes": fontes,
    }


def listar_historico(db: Session, limite: int = 20) -> List[Dict]:
    consultas = (
        db.query(Consulta)
        .order_by(Consulta.data_hora.desc())
        .limit(limite)
        .all()
    )
    return [
        {
            "id": c.id,
            "pergunta": c.pergunta,
            "data_hora": c.data_hora.isoformat(),
            "resposta": c.resposta.texto_resposta if c.resposta else None,
        }
        for c in consultas
    ]
