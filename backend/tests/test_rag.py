"""
Testes automatizados do pipeline RAG.

Execução:
    cd backend
    pytest tests/ -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from app.services.ingestao_service import _dividir_em_chunks


# ─────────────────────────────────────────────
# Testes de Chunking
# ─────────────────────────────────────────────

class TestChunking:
    def test_texto_curto_retorna_um_chunk(self):
        texto = "Este é um texto curto para teste."
        chunks = _dividir_em_chunks(texto)
        assert len(chunks) >= 1
        assert chunks[0] == texto

    def test_texto_longo_gera_multiplos_chunks(self):
        # Gera texto maior que CHUNK_SIZE (512 chars)
        paragrafo = "Dispositivo legal aplicável ao caso. " * 20
        texto = "\n\n".join([paragrafo] * 5)
        chunks = _dividir_em_chunks(texto)
        assert len(chunks) > 1

    def test_chunks_nao_sao_vazios(self):
        texto = "\n\n".join(["Parágrafo jurídico número {}.".format(i) for i in range(30)])
        chunks = _dividir_em_chunks(texto)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_preserva_conteudo(self):
        """Todo conteúdo do texto deve aparecer em algum chunk."""
        palavras_chave = ["STJ", "aposentadoria", "insalubridade"]
        texto = "\n\n".join([
            "O STJ decidiu sobre o caso de aposentadoria especial.",
            "A insalubridade foi comprovada por laudo técnico.",
            "O recurso foi provido por unanimidade.",
        ])
        chunks = _dividir_em_chunks(texto)
        texto_chunks = " ".join(chunks)
        for palavra in palavras_chave:
            assert palavra in texto_chunks

    def test_paragrafo_longo_nao_duplica_buffer(self):
        """
        Regressão: após um parágrafo maior que o chunk, o buffer anterior
        era reaproveitado e aparecia duplicado (e fora de ordem) no chunk seguinte.
        """
        from app.services.ingestao_service import settings
        curto_a = "Art. 1º Primeiro dispositivo."
        longo   = "B" * (settings.chunk_size + 100)
        curto_c = "Art. 3º Terceiro dispositivo."
        chunks = _dividir_em_chunks("\n\n".join([curto_a, longo, curto_c]))

        assert sum(curto_a in c for c in chunks) == 1
        assert chunks[0] == curto_a
        assert chunks[-1] == curto_c


# ─────────────────────────────────────────────
# Testes de Texto Legal (redação vigente e chunking por artigo)
# ─────────────────────────────────────────────

# Formato do texto compilado do Planalto: todas as redações em ordem cronológica
LEI_EXEMPLO = """LEI Nº 1, DE 1 DE JANEIRO DE 2000

Dispõe sobre testes.

CAPÍTULO I

Das Disposições Gerais

Art. 1º Texto original do artigo primeiro.

Art. 1º Texto vigente do artigo primeiro.     (Redação dada pela Lei nº 2, de 2001)

Art. 2º São dependentes:

I - o cônjuge;

I - o cônjuge e o companheiro;     (Redação dada pela Lei nº 3, de 2002)

II - os pais;

III - a pessoa designada.     (Revogada pela Lei nº 4, de 2003)

§ 1º O inciso abaixo é do parágrafo:

I - inciso do parágrafo;

Art. 3º Abono anual.

Parágrafo único. Versão original.

Parágrafo único. Versão da medida provisória:     (Vigência encerrada)

I - inciso trazido pela medida provisória;

Parágrafo único. Versão original.

Art. 4º Artigo revogado inteiro.     (Revogado pela Lei nº 6, de 2004)

§ 1º Parágrafo do artigo revogado.     (Revogado pela Lei nº 6, de 2004)

Art. 5º Esta Lei entra em vigor na data de sua publicação,
continuando em outra linha.

Brasília, 1 de janeiro de 2000.

FULANO DE TAL
"""


class TestTextoLegal:
    @pytest.fixture
    def artigos(self):
        from app.services.texto_legal import extrair_artigos
        artigos, _ = extrair_artigos(LEI_EXEMPLO)
        return {a[0].split()[1]: a for a in artigos}

    def test_reconhece_texto_legal(self):
        from app.services.texto_legal import eh_texto_legal
        assert eh_texto_legal(LEI_EXEMPLO)
        assert not eh_texto_legal("Um parecer qualquer.\n\nSem artigos.")

    def test_mantem_so_a_redacao_vigente(self, artigos):
        assert artigos["1º"] == [
            "Art. 1º Texto vigente do artigo primeiro. (Redação dada pela Lei nº 2, de 2001)"
        ]

    def test_revogado_vira_rotulo_e_nota(self, artigos):
        assert "III - (Revogada pela Lei nº 4, de 2003)" in artigos["2º"]

    def test_incisos_de_pais_diferentes_nao_colidem(self, artigos):
        assert artigos["2º"] == [
            "Art. 2º São dependentes:",
            "I - o cônjuge e o companheiro; (Redação dada pela Lei nº 3, de 2002)",
            "II - os pais;",
            "III - (Revogada pela Lei nº 4, de 2003)",
            "§ 1º O inciso abaixo é do parágrafo:",
            "I - inciso do parágrafo;",
        ]

    def test_redacao_substituida_leva_os_filhos(self, artigos):
        # a redação da MP (com inciso) perdeu a vigência e a original voltou
        assert artigos["3º"] == ["Art. 3º Abono anual.", "Parágrafo único. Versão original."]

    def test_artigo_revogado_descarta_os_filhos(self, artigos):
        assert artigos["4º"] == ["Art. 4º (Revogado pela Lei nº 6, de 2004)"]

    def test_continuacao_titulos_e_epilogo(self, artigos):
        from app.services.texto_legal import extrair_artigos
        _, fora = extrair_artigos(LEI_EXEMPLO)
        assert artigos["5º"] == [
            "Art. 5º Esta Lei entra em vigor na data de sua publicação, continuando em outra linha."
        ]
        tudo = "\n".join(l for a in artigos.values() for l in a)
        assert "CAPÍTULO" not in tudo and "Disposições Gerais" not in tudo
        assert "FULANO" not in tudo
        assert "Dispõe sobre testes." in fora

    def test_chunks_nao_misturam_artigos(self):
        import re
        chunks = _dividir_em_chunks(LEI_EXEMPLO)
        for chunk in chunks:
            assert len(re.findall(r"^Art\. \d", chunk, re.M)) <= 1
        assert "Art. 1º Texto original do artigo primeiro." not in "\n".join(chunks)

    def test_artigo_longo_repete_o_caput(self):
        from app.services.texto_legal import dividir_artigo
        caput = "Art. 9º São requisitos do benefício:"
        incisos = [f"{n} - requisito número {n} " + "x" * 90 for n in "I II III IV V VI VII VIII".split()]
        chunks = dividir_artigo([caput] + incisos, 512, 50)

        assert len(chunks) > 1
        assert all(len(c) <= 512 for c in chunks)
        assert all(c.startswith(caput) for c in chunks)
        for inciso in incisos:
            assert sum(inciso in c for c in chunks) == 1

    def test_dispositivo_maior_que_o_chunk_e_fatiado(self):
        from app.services.texto_legal import dividir_artigo
        caput = "Art. 9º Caput curto."
        longo = "§ 1º " + " ".join(f"palavra{i}" for i in range(300))
        chunks = dividir_artigo([caput, longo], 512, 50)

        assert all(len(c) <= 512 for c in chunks)
        assert all(c.startswith(caput) for c in chunks)
        assert "palavra299" in chunks[-1]
        # o corte acontece entre palavras
        assert all(c.split("\n", 1)[1].split()[0].startswith(("§", "palavra")) for c in chunks)

    def test_juntar_artigo_desfaz_a_divisao(self):
        from app.services.texto_legal import dividir_artigo, juntar_artigo, rotulo_do_artigo
        caput = "Art. 10. São requisitos do benefício:"
        incisos = [f"{n} - requisito número {n} " + "x" * 90 for n in "I II III IV V VI".split()]
        chunks = dividir_artigo([caput] + incisos, 512, 50)

        assert len(chunks) > 1
        assert {rotulo_do_artigo(c) for c in chunks} == {"Art. 10."}
        assert juntar_artigo(chunks) == "\n".join([caput] + incisos)

    def test_rotulo_do_artigo(self):
        from app.services.texto_legal import rotulo_do_artigo
        assert rotulo_do_artigo("Art. 29-A. O INSS ...") == "Art. 29-A."
        assert rotulo_do_artigo("Art. 152 (Revogado pela Lei nº 9.528, de 1997)") == "Art. 152"
        assert rotulo_do_artigo("Parágrafo solto") is None

    def test_lei_8213_art_16_so_com_redacao_vigente(self):
        from pathlib import Path
        arquivo = (Path(__file__).resolve().parents[2]
                   / "leis" / "lei_8213_91.txt")
        if not arquivo.exists():
            pytest.skip("arquivo da lei ausente")

        chunks = _dividir_em_chunks(arquivo.read_text(encoding="utf-8"))
        art16 = [c for c in chunks if c.startswith("Art. 16.")]
        texto = "\n".join(art16)

        assert art16
        assert "ou deficiência grave" in texto
        assert "IV - (Revogada pela Lei nº 9.032, de 1995)" in texto
        # redação original do inciso I (de 1991) não pode sobrar
        assert "filho, de qualquer condição, menor de 21 (vinte e um) anos ou inválido;" not in texto
        # o caput não fica mais colado no fim do Art. 15
        assert not any("Art. 15" in c and "Art. 16." in c for c in chunks)


# ─────────────────────────────────────────────
# Testes de Leitura de Arquivo
# ─────────────────────────────────────────────

class TestLeituraArquivo:
    def test_ler_txt(self, tmp_path):
        from app.services.ingestao_service import ler_arquivo
        arquivo = tmp_path / "teste.txt"
        arquivo.write_text("Conteúdo jurídico de teste.", encoding="utf-8")
        texto = ler_arquivo(str(arquivo))
        assert "Conteúdo jurídico" in texto

    def test_formato_nao_suportado(self, tmp_path):
        from app.services.ingestao_service import ler_arquivo
        arquivo = tmp_path / "teste.docx"
        arquivo.write_bytes(b"fake content")
        with pytest.raises(ValueError, match="Formato não suportado"):
            ler_arquivo(str(arquivo))

    def test_ler_html(self, tmp_path):
        from app.services.ingestao_service import ler_arquivo
        arquivo = tmp_path / "teste.html"
        arquivo.write_text(
            "<html><body><h1>Acórdão</h1><p>Texto da decisão.</p></body></html>",
            encoding="utf-8"
        )
        texto = ler_arquivo(str(arquivo))
        assert "Acórdão" in texto
        assert "Texto da decisão" in texto


# ─────────────────────────────────────────────
# Testes de Integração (com mock do banco)
# ─────────────────────────────────────────────

class TestBuscaVetorial:
    def _buscar(self, id_documento=None):
        import numpy as np
        from app.services.ingestao_service import buscar_chunks

        chunk_mock = MagicMock()
        db_mock = MagicMock()
        # <#> devolve o produto interno negativo
        db_mock.execute.return_value.all.return_value = [(chunk_mock, -0.82)]

        with patch("app.services.ingestao_service.gerar_embeddings",
                   return_value=np.zeros((1, 1536), dtype="float32")):
            resultado = buscar_chunks(db_mock, "pergunta", top_k=3,
                                      id_documento=id_documento)

        from sqlalchemy.dialects import postgresql
        stmt = db_mock.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        return resultado, chunk_mock, sql

    def test_usa_produto_interno_do_pgvector(self):
        resultado, chunk_mock, sql = self._buscar()
        assert "<#>" in sql
        assert "LIMIT" in sql
        assert "chunks.id_documento =" not in sql
        # score volta positivo, como no antigo IndexFlatIP
        assert resultado == [(chunk_mock, pytest.approx(0.82))]

    def test_filtra_por_documento_no_banco(self):
        _, _, sql = self._buscar(id_documento=7)
        assert "chunks.id_documento =" in sql


class TestRagService:
    def test_recuperar_chunks_repassa_documento_e_scores(self):
        from app.services import rag_service

        chunk = MagicMock()
        chunk.id = 10
        chunk.id_documento = 3
        chunk.texto_chunk = "Parecer sem artigos."
        chunk.documento.titulo = "Lei 8.213/91"
        chunk.documento.fonte.nome = "Planalto"

        with patch("app.services.rag_service.buscar_chunks",
                   return_value=[(chunk, 0.9)]) as busca:
            fontes = rag_service._recuperar_chunks(MagicMock(), "dependentes", 5, id_documento=3)

        assert busca.call_args.kwargs["id_documento"] == 3
        assert fontes == [{
            "chunk_id": 10,
            "titulo": "Lei 8.213/91",
            "fonte": "Planalto",
            "texto_chunk": "Parecer sem artigos.",
            "score": 0.9,
        }]

    def _chunk(self, id_, texto):
        chunk = MagicMock()
        chunk.id, chunk.id_documento, chunk.texto_chunk = id_, 3, texto
        return chunk

    def test_recuperar_chunks_entrega_o_artigo_inteiro_uma_vez(self):
        from app.services import rag_service

        caput = "Art. 16. São beneficiários, na condição de dependentes do segurado:"
        artigo = f"{caput}\nI - o cônjuge;\n§ 5º Prova de união estável."
        resultados = [
            (self._chunk(1, f"{caput}\n§ 5º Prova de união estável."), 0.8),
            (self._chunk(2, f"{caput}\nI - o cônjuge;"), 0.7),
            (self._chunk(3, "Art. 17. Outro artigo."), 0.6),
        ]
        with patch("app.services.rag_service.buscar_chunks", return_value=resultados), \
             patch("app.services.rag_service.texto_do_artigo",
                   side_effect=lambda db, doc, rotulo: artigo if rotulo == "Art. 16." else "Art. 17. Outro artigo.") as remonta:
            fontes = rag_service._recuperar_chunks(MagicMock(), "dependentes", 5)

        assert [f["texto_chunk"] for f in fontes] == [artigo, "Art. 17. Outro artigo."]
        assert [f["score"] for f in fontes] == [0.8, 0.6]
        assert remonta.call_args_list[0].args[1:] == (3, "Art. 16.")

    def test_artigo_grande_demais_fica_so_o_chunk(self):
        from app.services import rag_service

        chunk = self._chunk(1, "Art. 11. São segurados obrigatórios:\nI - como empregado;")
        with patch("app.services.rag_service.buscar_chunks", return_value=[(chunk, 0.8)]), \
             patch("app.services.rag_service.texto_do_artigo", return_value="x" * 10_000):
            fontes = rag_service._recuperar_chunks(MagicMock(), "segurados", 5)

        assert fontes[0]["texto_chunk"] == chunk.texto_chunk

    def test_responder_sem_documentos_indexados(self):
        """Sem chunks indexados, deve retornar mensagem de aviso."""
        from app.services import rag_service
        from app.services.ingestao_service import buscar_chunks

        db_mock = MagicMock()
        # Consulta salva com flush
        consulta_mock = MagicMock()
        consulta_mock.id = 1
        resposta_mock  = MagicMock()
        resposta_mock.id = 1
        db_mock.query.return_value.filter.return_value.all.return_value = []

        with patch("app.services.rag_service.buscar_chunks", return_value=[]):
            with patch("app.services.rag_service.Consulta", return_value=consulta_mock):
                with patch("app.services.rag_service.Resposta", return_value=resposta_mock):
                    resultado = rag_service.responder_pergunta(
                        db=db_mock,
                        pergunta="Quais decisões tratam de aposentadoria especial?",
                    )

        assert "consulta_id" in resultado
        assert "resposta" in resultado
        assert "indexe" in resultado["resposta"].lower() or "não encontrei" in resultado["resposta"].lower()

    def test_historico_formato(self):
        """listar_historico deve retornar lista de dicts com campos esperados."""
        from app.services.rag_service import listar_historico
        from datetime import datetime

        consulta_mock = MagicMock()
        consulta_mock.id = 1
        consulta_mock.pergunta = "Teste?"
        consulta_mock.data_hora = datetime(2026, 1, 15, 10, 30)
        consulta_mock.resposta = MagicMock()
        consulta_mock.resposta.texto_resposta = "Resposta de teste."

        db_mock = MagicMock()
        db_mock.query.return_value.order_by.return_value.limit.return_value.all.return_value = [
            consulta_mock
        ]

        historico = listar_historico(db=db_mock, limite=5)
        assert isinstance(historico, list)
        assert len(historico) == 1
        assert historico[0]["pergunta"] == "Teste?"
        assert "data_hora" in historico[0]


# ─────────────────────────────────────────────
# Testes de Hash de Integridade
# ─────────────────────────────────────────────

class TestHashIntegridade:
    def test_hash_consistente(self):
        import hashlib
        texto = "Documento jurídico de referência."
        h1 = hashlib.sha256(texto.encode()).hexdigest()
        h2 = hashlib.sha256(texto.encode()).hexdigest()
        assert h1 == h2

    def test_hash_diferente_para_textos_distintos(self):
        import hashlib
        h1 = hashlib.sha256("Texto A".encode()).hexdigest()
        h2 = hashlib.sha256("Texto B".encode()).hexdigest()
        assert h1 != h2


# ─────────────────────────────────────────────
# Testes das Rotas da API (TestClient)
# ─────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_root_responde(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_consulta_sem_pergunta_retorna_400(self, client):
        with patch("app.api.routes.rag_service.responder_pergunta"):
            r = client.post("/api/v1/consultar", json={"pergunta": ""})
        assert r.status_code == 400

    def test_consulta_com_pergunta_valida(self, client):
        mock_resultado = {
            "consulta_id": 1,
            "resposta": "Resposta de teste.",
            "fontes": [],
        }
        with patch("app.api.routes.rag_service.responder_pergunta", return_value=mock_resultado):
            r = client.post(
                "/api/v1/consultar",
                json={"pergunta": "O que é aposentadoria especial?"},
            )
        assert r.status_code == 200
        assert "resposta" in r.json()

    def test_rotas_de_avaliacao_removidas(self, client):
        assert client.post("/api/v1/comparar", json={"pergunta": "x"}).status_code == 404
        assert client.post("/api/v1/avaliar", json={"id_resposta": 1, "nota": 5}).status_code == 404
        assert client.get("/api/v1/resultados").status_code == 404
        assert client.get("/api/v1/resultados/csv").status_code == 404

    def test_provedor_desconhecido_cai_no_padrao(self):
        from app.services import rag_service
        assert "deepseek" not in rag_service.PROVEDORES
        with patch.object(rag_service, "_recuperar_chunks", return_value=[]),              patch.object(rag_service, "Consulta"), patch.object(rag_service, "Resposta"):
            r = rag_service.responder_pergunta(MagicMock(), "pergunta", provedor="deepseek")
        assert r["modelo_llm"] == "maritaca"

    def test_ingerir_responde_202_e_agenda_a_indexacao(self, client):
        doc = MagicMock(id=7, titulo="Lei X", status="processando")
        with patch("app.api.routes.ingestao_service.registrar_documento",
                   return_value=(doc, True)), \
             patch("app.api.routes.ingestao_service.processar_documento") as processar:
            r = client.post(
                "/api/v1/ingerir",
                files={"arquivo": ("lei.txt", b"Art. 1o Texto.", "text/plain")},
                data={"titulo": "Lei X"},
            )
        assert r.status_code == 202
        assert r.json()["status"] == "processando"
        processar.assert_called_once_with(7)

    def test_ingerir_arquivo_repetido_nao_reindexa(self, client):
        doc = MagicMock(id=7, titulo="Lei X", status="concluido")
        with patch("app.api.routes.ingestao_service.registrar_documento",
                   return_value=(doc, False)), \
             patch("app.api.routes.ingestao_service.processar_documento") as processar:
            r = client.post(
                "/api/v1/ingerir",
                files={"arquivo": ("lei.txt", b"Art. 1o Texto.", "text/plain")},
                data={"titulo": "Lei X"},
            )
        assert r.status_code == 202
        processar.assert_not_called()

    def test_token_exigido_quando_configurado(self, client):
        config = MagicMock(api_token="segredo")
        with patch("app.core.seguranca.get_settings", return_value=config), \
             patch("app.api.routes.rag_service.listar_historico", return_value=[]):
            assert client.get("/api/v1/historico").status_code == 401
            assert client.get("/api/v1/historico",
                              headers={"X-API-Key": "errado"}).status_code == 401
            assert client.get("/api/v1/historico",
                              headers={"X-API-Key": "segredo"}).status_code == 200
            # a página em / continua pública (ela é quem pede o token)
            assert client.get("/").status_code == 200

    def test_sem_token_configurado_acesso_livre(self, client):
        with patch("app.core.seguranca.get_settings", return_value=MagicMock(api_token="")), \
             patch("app.api.routes.rag_service.listar_historico", return_value=[]):
            assert client.get("/api/v1/historico").status_code == 200


# ─────────────────────────────────────────────
# Testes de Embeddings por API
# ─────────────────────────────────────────────

class TestEmbeddings:
    def _cliente(self, dim):
        def criar(model, input):
            resposta = MagicMock()
            resposta.data = [MagicMock(embedding=[3.0, 4.0] + [0.0] * (dim - 2)) for _ in input]
            return resposta
        cliente = MagicMock()
        cliente.embeddings.create.side_effect = criar
        return cliente

    def test_gera_em_lotes_e_normaliza(self):
        from app.services import ingestao_service as ing
        cliente = self._cliente(ing.settings.embedding_dim)
        with patch.object(ing, "_get_cliente_embeddings", return_value=cliente):
            vetores = ing.gerar_embeddings(["trecho"] * 300)

        assert vetores.shape == (300, ing.settings.embedding_dim)
        assert cliente.embeddings.create.call_count == 3      # lotes de 128
        assert list(vetores[0][:2]) == pytest.approx([0.6, 0.8])  # norma 1

    def test_dimensao_diferente_da_coluna_falha(self):
        from app.services import ingestao_service as ing
        with patch.object(ing, "_get_cliente_embeddings", return_value=self._cliente(384)):
            with pytest.raises(ValueError, match="EMBEDDING_DIM"):
                ing.gerar_embeddings(["trecho"])

    def test_lista_vazia_nao_chama_a_api(self):
        from app.services import ingestao_service as ing
        with patch.object(ing, "_get_cliente_embeddings") as cliente:
            assert ing.gerar_embeddings([]).shape == (0, ing.settings.embedding_dim)
        cliente.assert_not_called()


# ─────────────────────────────────────────────
# Testes da Ingestão em segundo plano
# ─────────────────────────────────────────────

class TestIngestaoEmSegundoPlano:
    def _arquivo(self, tmp_path):
        arquivo = tmp_path / "lei.txt"
        arquivo.write_text("Art. 1º Texto da lei.", encoding="utf-8")
        return str(arquivo)

    def _db(self, existente=None):
        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = existente
        return db

    def test_documento_novo_fica_processando(self, tmp_path):
        from app.services.ingestao_service import registrar_documento
        db = self._db()
        doc, processar = registrar_documento(db, self._arquivo(tmp_path), "Lei X")

        assert processar is True
        assert doc.status == "processando"
        db.add.assert_called_once_with(doc)
        db.commit.assert_called_once()

    def test_arquivo_ja_indexado_nao_reprocessa(self, tmp_path):
        from app.services.ingestao_service import registrar_documento
        existente = MagicMock(status="concluido")
        doc, processar = registrar_documento(self._db(existente), self._arquivo(tmp_path), "Lei X")
        assert (doc, processar) == (existente, False)

    def test_arquivo_que_falhou_e_reprocessado(self, tmp_path):
        from app.services.ingestao_service import registrar_documento
        existente = MagicMock(status="erro", erro="sem chave")
        doc, processar = registrar_documento(self._db(existente), self._arquivo(tmp_path), "Lei X")
        assert processar is True
        assert (doc.status, doc.erro) == ("processando", None)

    def _processar(self, embeddings):
        from app.services import ingestao_service as ing
        doc = MagicMock(id=1, titulo="Lei X", texto="Texto curto de um parecer.")
        db = MagicMock()
        db.get.return_value = doc
        with patch.object(ing, "SessionLocal", return_value=db), \
             patch.object(ing, "gerar_embeddings", **embeddings):
            ing.processar_documento(1)
        return doc, db

    def test_processar_grava_chunks_e_conclui(self):
        import numpy as np
        from app.models.models import Chunk
        from app.services.ingestao_service import settings
        doc, db = self._processar(
            {"return_value": np.zeros((1, settings.embedding_dim), dtype="float32")})

        assert doc.status == "concluido"
        assert isinstance(db.add.call_args.args[0], Chunk)
        db.commit.assert_called_once()
        db.close.assert_called_once()

    def test_falha_na_api_marca_erro(self):
        doc, db = self._processar({"side_effect": RuntimeError("chave inválida")})

        db.rollback.assert_called_once()
        assert doc.status == "erro"
        assert "chave inválida" in doc.erro
        db.close.assert_called_once()


# ─────────────────────────────────────────────
# Testes de erros de provedor, histórico e gestão de documentos
# ─────────────────────────────────────────────

class TestErrosDeProvedor:
    def test_falha_do_llm_vira_erro_provedor(self):
        from app.services import rag_service
        fontes = [{"chunk_id": 1, "titulo": "Lei", "fonte": "x", "texto_chunk": "Art. 1º", "score": .9}]
        llm = MagicMock(side_effect=RuntimeError("401 da API"))
        with patch.object(rag_service, "_recuperar_chunks", return_value=fontes), \
             patch.object(rag_service, "_get_llm", return_value=llm):
            with pytest.raises(rag_service.ErroProvedor, match="Maritaca"):
                rag_service.responder_pergunta(MagicMock(), "pergunta")

    def test_falha_nos_embeddings_vira_erro_provedor(self):
        from openai import APIConnectionError
        from app.services import rag_service
        erro = APIConnectionError(request=MagicMock())
        with patch.object(rag_service, "_recuperar_chunks", side_effect=erro):
            with pytest.raises(rag_service.ErroProvedor, match="embeddings"):
                rag_service.responder_pergunta(MagicMock(), "pergunta")

    def test_falha_nao_grava_no_historico(self):
        from app.services import rag_service
        db = MagicMock()
        with patch.object(rag_service, "_recuperar_chunks", return_value=[{"texto_chunk": "x"}]), \
             patch.object(rag_service, "_get_llm", return_value=MagicMock(side_effect=RuntimeError())):
            with pytest.raises(rag_service.ErroProvedor):
                rag_service.responder_pergunta(db, "pergunta")
        db.add.assert_not_called()
        db.commit.assert_not_called()


class TestObterConsulta:
    def test_mantem_a_ordem_e_marca_trechos_indisponiveis(self):
        from datetime import datetime
        from app.services import rag_service
        chunk = MagicMock(id=5, texto_chunk="Parecer sem artigo.")
        chunk.documento.titulo = "Parecer"
        chunk.documento.fonte.nome = "AGU"
        usados = [MagicMock(id=2, chunk=None, score=.5), MagicMock(id=1, chunk=chunk, score=.9)]
        consulta = MagicMock(id=3, pergunta="P?", data_hora=datetime(2026, 1, 1))
        consulta.resposta.texto_resposta = "R [1]"
        consulta.resposta.modelo_llm = "openai"
        consulta.resposta.documentos_utilizados = usados
        db = MagicMock()
        db.get.return_value = consulta

        r = rag_service.obter_consulta(db, 3)

        assert r["resposta"] == "R [1]" and r["modelo_llm"] == "openai"
        assert [f["titulo"] for f in r["fontes"]] == ["Parecer", "Trecho indisponível"]
        assert r["fontes"][0]["texto_chunk"] == "Parecer sem artigo."

    def test_consulta_inexistente(self):
        from app.services import rag_service
        db = MagicMock()
        db.get.return_value = None
        assert rag_service.obter_consulta(db, 99) is None


class TestRotasDeDocumentos:
    @pytest.fixture
    def db(self):
        from app.core.database import get_db
        from app.main import app
        db = MagicMock()
        app.dependency_overrides[get_db] = lambda: db
        yield db
        app.dependency_overrides.clear()

    @pytest.fixture
    def client(self, db):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_consulta_com_llm_fora_do_ar_retorna_502(self, client):
        from app.services.rag_service import ErroProvedor
        with patch("app.api.routes.rag_service.responder_pergunta",
                   side_effect=ErroProvedor("O modelo Maritaca não respondeu.")):
            r = client.post("/api/v1/consultar", json={"pergunta": "x"})
        assert r.status_code == 502
        assert "Maritaca" in r.json()["detail"]

    def test_detalhe_do_historico(self, client):
        with patch("app.api.routes.rag_service.obter_consulta", return_value={"consulta_id": 1}):
            assert client.get("/api/v1/historico/1").json() == {"consulta_id": 1}
        with patch("app.api.routes.rag_service.obter_consulta", return_value=None):
            assert client.get("/api/v1/historico/2").status_code == 404

    def test_lista_documentos_com_total_de_chunks(self, client, db):
        doc = MagicMock(id=1, titulo="Lei", status="concluido", erro=None, criado_em=None)
        db.query.return_value.order_by.return_value.all.return_value = [doc]
        db.query.return_value.group_by.return_value.all.return_value = [(1, 42)]
        r = client.get("/api/v1/documentos")
        assert r.json()[0]["total_chunks"] == 42

    def test_excluir_documento(self, client, db):
        doc = MagicMock(id=1, status="concluido")
        db.get.return_value = doc
        with patch("app.api.routes.ingestao_service.excluir_documento") as excluir:
            r = client.delete("/api/v1/documentos/1")
        assert r.status_code == 204
        excluir.assert_called_once_with(db, doc)

    def test_nao_mexe_em_documento_processando(self, client, db):
        db.get.return_value = MagicMock(id=1, status="processando")
        with patch("app.api.routes.ingestao_service.excluir_documento") as excluir:
            assert client.delete("/api/v1/documentos/1").status_code == 409
            assert client.post("/api/v1/documentos/1/reindexar").status_code == 409
        excluir.assert_not_called()

    def test_documento_inexistente(self, client, db):
        db.get.return_value = None
        assert client.delete("/api/v1/documentos/9").status_code == 404

    def test_reindexar_agenda_o_processamento(self, client, db):
        doc = MagicMock(id=4, titulo="Lei", status="erro")
        db.get.return_value = doc
        with patch("app.api.routes.ingestao_service.preparar_reindexacao") as preparar, \
             patch("app.api.routes.ingestao_service.processar_documento") as processar:
            r = client.post("/api/v1/documentos/4/reindexar")
        assert r.status_code == 202
        preparar.assert_called_once_with(db, doc)
        processar.assert_called_once_with(4)


class TestExclusaoEReindexacao:
    def test_excluir_apaga_e_confirma(self):
        from app.services.ingestao_service import excluir_documento
        db = MagicMock()
        excluir_documento(db, MagicMock(id=1))
        # solta respostas, apaga chunks, solta consultas, apaga o documento
        assert db.execute.call_count == 4
        db.commit.assert_called_once()

    def test_reindexar_volta_a_processando(self):
        from app.services.ingestao_service import preparar_reindexacao
        db = MagicMock()
        doc = MagicMock(id=1, status="erro", erro="sem chave")
        preparar_reindexacao(db, doc)
        assert (doc.status, doc.erro) == ("processando", None)
        db.commit.assert_called_once()


class TestSugestoes:
    def test_pergunta_sobre_artigo(self):
        from app.services.rag_service import _pergunta_sobre
        assert _pergunta_sobre("Lei 8.742/93", "Art. 20. O benefício...") == \
            "O que diz o Art. 20 da Lei 8.742/93?"
        assert _pergunta_sobre("Decreto 3.048/99", "Art. 1o Texto") == \
            "O que diz o Art. 1º do Decreto 3.048/99?"
        assert _pergunta_sobre("Parecer AGU", "Texto sem artigo") is None

    def _gerar(self, linhas, limite=3):
        from app.services.rag_service import gerar_sugestoes
        db = MagicMock()
        db.execute.return_value.all.return_value = linhas
        return gerar_sugestoes(db, limite)

    def test_intercala_documentos(self):
        linhas = [
            ("Art. 1º a", 1, "Lei A"), ("Art. 2º b", 1, "Lei A"),
            ("Art. 3º c", 1, "Lei A"), ("Art. 9º d", 2, "Lei B"),
        ]
        r = self._gerar(linhas)
        assert [s["id_documento"] for s in r] == [1, 2, 1]
        assert r[1]["pergunta"] == "O que diz o Art. 9º da Lei B?"

    def test_lei_nao_gera_pergunta_generica(self):
        r = self._gerar([("Brasília, 8 de maio", 1, "Lei A"), ("Art. 1º a", 1, "Lei A")])
        assert [s["pergunta"] for s in r] == ["O que diz o Art. 1º da Lei A?"]

    def test_documento_sem_artigos_gera_uma_pergunta_geral(self):
        r = self._gerar([("parágrafo um", 5, "Parecer X"), ("parágrafo dois", 5, "Parecer X")])
        assert r == [{"pergunta": "Quais são os principais pontos do Parecer X?", "id_documento": 5}]

    def test_sem_documentos(self):
        assert self._gerar([]) == []

    def test_rota_limita_a_quantidade(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with patch("app.api.routes.rag_service.gerar_sugestoes", return_value=[]) as gerar:
            assert TestClient(app).get("/api/v1/sugestoes?limite=50").json() == []
        assert gerar.call_args.kwargs["limite"] == 10
