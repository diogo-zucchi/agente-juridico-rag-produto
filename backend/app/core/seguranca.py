"""
Acesso à API: com API_TOKEN definido, as rotas /api/v1 exigem o cabeçalho
X-API-Key. Cada consulta gasta créditos do LLM e o upload grava no banco,
então uma URL pública não deve ficar aberta.
"""
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_cabecalho = APIKeyHeader(name="X-API-Key", auto_error=False)


def exigir_token(token: str | None = Security(_cabecalho)) -> None:
    esperado = get_settings().api_token
    if not esperado:
        return  # sem token configurado (desenvolvimento local): acesso livre
    if not token or not secrets.compare_digest(token, esperado):
        raise HTTPException(status_code=401, detail="Token de acesso inválido ou ausente.")
