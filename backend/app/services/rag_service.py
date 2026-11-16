import re
from typing import List, Dict, Any, Callable, Optional, Tuple
from openai import OpenAI, OpenAIError
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.models import (
    STATUS_CONCLUIDO, Chunk, Consulta, Documento, Resposta, RespostaDocumento,
)
from app.services.ingestao_service import buscar_artigos, buscar_chunks, texto_do_artigo
from app.services.texto_legal import (
    artigos_citados, chave_do_artigo, rotulo_do_artigo,
)

PROMPT_JURIDICO = """Você é um assistente jurídico especializado no direito brasileiro.
Utilize EXCLUSIVAMENTE os trechos de documentos jurídicos fornecidos abaixo.
Nunca invente informações. Se a resposta não estiver nos documentos, diga claramente.
Se a pergunta se referir a uma lei específica, use apenas os documentos dessa lei.
Sempre indique as fontes utilizadas, citando o número do documento entre colchetes, como [1].

=== DOCUMENTOS JURÍDICOS RECUPERADOS ===
{contexto}
=========================================

Pergunta: {pergunta}

Resposta fundamentada:"""

# Provedores de LLM suportados
PROVEDORES = ("maritaca", "openai")
NOMES_PROVEDORES = {"maritaca": "Maritaca", "openai": "OpenAI"}


class ErroProvedor(Exception):
    """Falha de um serviço externo (LLM ou embeddings). A rota responde 502
    com esta mensagem, e nada é gravado no histórico."""

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

    Artigos citados pelo número na pergunta ("Art. 21", "arts. 20 e 21")
    entram primeiro, com o texto exato (ver _artigos_citados): o número não
    tem significado para o embedding. A busca semântica completa as vagas.
    """
    max_chars = get_settings().artigo_max_chars
    fontes: List[Dict] = []
    vistos = set()

    def incluir(chunk: Chunk, score: float, exato: bool) -> bool:
        """Acrescenta a fonte se for nova; devolve True quando as k vagas acabam."""
        texto, chave = _texto_expandido(db, chunk, max_chars)
        if chave not in vistos:
            vistos.add(chave)
            fontes.append(_descrever_fonte(chunk, texto, score, exato))
        return len(fontes) == k

    for chunk in _artigos_citados(db, pergunta, k, id_documento):
        if incluir(chunk, SCORE_EXATO, exato=True):
            return fontes

    # busca mais candidatos, porque vários podem ser do mesmo artigo
    resultados = buscar_chunks(db, pergunta, top_k=k * 3, id_documento=id_documento)
    for chunk, score in resultados:
        if incluir(chunk, score, exato=False):
            break
    return fontes


# ─────────────────────────────────────────────
# Artigos citados pelo número
# ─────────────────────────────────────────────
# score do artigo citado: o máximo da escala, acima de qualquer resultado
# semântico (produto interno entre embeddings normalizados, no máximo 1)
SCORE_EXATO = 1.0
# sem documento definido, quantas leis podem trazer o mesmo artigo
MAX_DOCUMENTOS_POR_ARTIGO = 3


def _artigos_citados(
    db: Session,
    pergunta: str,
    k: int,
    id_documento: int | None = None,
) -> List[Chunk]:
    """
    Primeiro chunk de cada artigo citado na pergunta, na ordem da citação.

    Com id_documento, o artigo é buscado só nele. Sem ele, vem o artigo de
    cada documento que o tiver (até MAX_DOCUMENTOS_POR_ARTIGO por artigo),
    cada um com o título da sua lei no contexto. Não há regra para adivinhar
    de qual lei a pergunta fala: quem escolhe é o LLM, que entende "LOAS",
    "Lei 8.742/93" ou "Lei de Benefícios" sem depender de como o título foi
    escrito. Artigo citado que não existe nos documentos é ignorado.
    """
    chaves = artigos_citados(pergunta)
    if not chaves:
        return []

    ids = [id_documento] if id_documento is not None else None
    por_chave: Dict[str, List[Chunk]] = {}
    for chave, chunk in buscar_artigos(db, chaves, ids):
        por_chave.setdefault(chave, []).append(chunk)

    limite = None if ids else MAX_DOCUMENTOS_POR_ARTIGO
    exatos = [chunk for chave in chaves for chunk in por_chave.get(chave, [])[:limite]]
    return exatos[:k]


def _texto_expandido(db: Session, chunk: Chunk, max_chars: int) -> Tuple[str, tuple]:
    """Texto entregue ao LLM: o artigo inteiro, se couber; senão, o chunk."""
    rotulo = rotulo_do_artigo(chunk.texto_chunk)
    if rotulo:
        artigo = texto_do_artigo(db, chunk.id_documento, rotulo)
        if artigo and len(artigo) <= max_chars:
            return artigo, ("artigo", chunk.id_documento, rotulo)
    return chunk.texto_chunk, ("chunk", chunk.id)


def _descrever_fonte(chunk: Chunk, texto: str, score: float, exato: bool = False) -> Dict:
    doc = chunk.documento
    return {
        "chunk_id": chunk.id,
        "titulo": doc.titulo if doc else "Documento",
        "fonte": doc.fonte.nome if doc and doc.fonte else "Fonte desconhecida",
        "texto_chunk": texto,
        "score": score,
        # artigo citado pelo número na pergunta (não veio da busca semântica)
        "exato": exato,
    }


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
        raise ErroProvedor(
            f"O modelo {NOMES_PROVEDORES[provedor]} não respondeu. "
            "Verifique a chave de API e o modelo configurado, ou tente o outro modelo."
        ) from e


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

    try:
        fontes = _recuperar_chunks(db, pergunta, k, id_documento)
    except OpenAIError as e:
        logger.error(f"Erro ao gerar o embedding da pergunta: {e}")
        raise ErroProvedor(
            "Não foi possível consultar o serviço de embeddings. "
            "Verifique a chave e o modelo de embeddings configurados."
        ) from e
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


def obter_consulta(db: Session, id_consulta: int) -> Optional[Dict[str, Any]]:
    """
    Uma consulta do histórico com a resposta salva e as fontes, na mesma
    ordem em que foram entregues ao LLM (é ela que dá sentido às citações [n]).
    Trechos que não existem mais (documento excluído ou reindexado) ficam
    no lugar como indisponíveis, para não desalinhar a numeração.
    """
    consulta = db.get(Consulta, id_consulta)
    if consulta is None:
        return None
    resposta = consulta.resposta
    max_chars = get_settings().artigo_max_chars
    citados = set(artigos_citados(consulta.pergunta or ""))

    fontes = []
    usados = sorted(resposta.documentos_utilizados, key=lambda u: u.id) if resposta else []
    for usado in usados:
        if usado.chunk is None:
            fontes.append({
                "chunk_id": None,
                "titulo": "Trecho indisponível",
                "fonte": "O documento foi excluído ou reindexado",
                "texto_chunk": "",
                "score": usado.score,
            })
            continue
        texto, _ = _texto_expandido(db, usado.chunk, max_chars)
        # o vínculo não guarda se a fonte foi um artigo citado: deduz pela
        # pergunta salva e pelo score do acerto exato
        exato = usado.score == SCORE_EXATO and chave_do_artigo(usado.chunk.texto_chunk) in citados
        fontes.append(_descrever_fonte(usado.chunk, texto, usado.score, exato))

    return {
        "consulta_id": consulta.id,
        "pergunta": consulta.pergunta,
        "data_hora": consulta.data_hora.isoformat(),
        "modelo_llm": resposta.modelo_llm if resposta else None,
        "resposta": resposta.texto_resposta if resposta else None,
        "fontes": fontes,
    }


# ─────────────────────────────────────────────
# Sugestões de perguntas (a partir dos documentos indexados)
# ─────────────────────────────────────────────
# trechos curtos costumam ser artigos revogados ou de vigência ("Esta Lei
# entra em vigor..."), que não rendem boas perguntas
_MIN_CHARS_SUGESTAO = 200
_FEMININOS = {"lei", "lc", "constituição", "constituicao", "emenda", "medida", "portaria",
              "instrução", "instrucao", "resolução", "resolucao", "súmula", "sumula",
              "norma", "consolidação", "consolidacao", "clt", "orientação", "orientacao"}
_MASCULINOS = {"decreto", "decreto-lei", "código", "codigo", "estatuto", "parecer",
               "regulamento", "provimento", "ato", "enunciado", "regimento"}


def _artigo_definido(titulo: str) -> str:
    """'da Lei…', 'do Decreto…' ou 'de …' conforme a primeira palavra do título."""
    primeira = (titulo.split() or [""])[0].lower().strip(".,:;")
    if primeira in _FEMININOS:
        return "da"
    if primeira in _MASCULINOS:
        return "do"
    return "de"


def _pergunta_sobre(titulo: str, trecho: str) -> Optional[str]:
    """Pergunta sobre o artigo no início do trecho, ou None se não houver artigo."""
    rotulo = rotulo_do_artigo(trecho)
    if not rotulo:
        return None
    # "Art. 1o" -> "Art. 1º" (textos do Planalto usam a letra o como ordinal)
    rotulo = re.sub(r"(\d)o\b", r"\1º", rotulo.rstrip(". "))
    return f"O que diz o {rotulo} {_artigo_definido(titulo)} {titulo}?"


def gerar_sugestoes(db: Session, limite: int = 3) -> List[Dict[str, Any]]:
    """
    Perguntas sobre trechos sorteados dos documentos já indexados (em leis,
    um artigo; nos demais, o documento todo), variando entre documentos.
    Cada sugestão leva o id do documento, para a consulta ser feita nele.
    """
    linhas = db.execute(
        select(Chunk.texto_chunk, Documento.id, Documento.titulo)
        .join(Documento, Chunk.id_documento == Documento.id)
        .where(Documento.status == STATUS_CONCLUIDO)
        .where(func.length(Chunk.texto_chunk) >= _MIN_CHARS_SUGESTAO)
        .order_by(func.random())
        .limit(limite * 15)
    ).all()

    por_documento: Dict[int, List[Dict[str, Any]]] = {}
    titulos: Dict[int, str] = {}
    vistas = set()
    for texto, id_documento, titulo in linhas:
        titulos[id_documento] = titulo
        pergunta = _pergunta_sobre(titulo, texto)
        if pergunta is None or pergunta in vistas:
            continue
        vistas.add(pergunta)
        por_documento.setdefault(id_documento, []).append(
            {"pergunta": pergunta, "id_documento": id_documento}
        )
    # documento sem artigos (parecer, decisão…): uma pergunta geral sobre ele
    for id_documento, titulo in titulos.items():
        if id_documento not in por_documento:
            por_documento[id_documento] = [{
                "pergunta": f"Quais são os principais pontos {_artigo_definido(titulo)} {titulo}?",
                "id_documento": id_documento,
            }]

    # intercala os documentos: uma sugestão de cada antes de repetir algum
    sugestoes: List[Dict[str, Any]] = []
    filas = list(por_documento.values())
    while filas and len(sugestoes) < limite:
        for fila in list(filas):
            sugestoes.append(fila.pop(0))
            if not fila:
                filas.remove(fila)
            if len(sugestoes) == limite:
                break
    return sugestoes
