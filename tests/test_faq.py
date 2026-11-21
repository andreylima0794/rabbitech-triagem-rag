"""python -m unittest -v"""

import tempfile
import unittest
import warnings
from pathlib import Path

from src import faq

RAIZ = Path(__file__).resolve().parent.parent
FAQ_REAL = RAIZ / "base" / "faq_interno.md"
CATEGORIAS_REAIS = RAIZ / "base" / "categorias.toml"


def _arquivo(pasta: Path, nome: str, texto: str) -> Path:
    caminho = pasta / nome
    caminho.write_text(texto, encoding="utf-8")
    return caminho


class TestFAQReal(unittest.TestCase):
    def setUp(self):
        self.faq = faq.carregar(FAQ_REAL, CATEGORIAS_REAIS, 18)

    def test_18_registros_com_id_estavel(self):
        self.assertEqual([r.id for r in self.faq.registros], list(range(1, 19)))
        self.assertTrue(self.faq.registros[0].pergunta.startswith("Como faço para redefinir minha senha"))
        self.assertEqual(self.faq.por_id(18).pergunta, "Onde vejo minhas faturas?")

    def test_revisao_e_hash(self):
        self.assertEqual(self.faq.revisao, "12/2025")
        self.assertEqual(len(self.faq.hash), 64)

    def test_categorias(self):
        self.assertEqual(self.faq.por_id(1).categorias, ("acesso_senha",))
        self.assertEqual(self.faq.por_id(6).categorias, ("usuarios_planos", "assinatura_produto"))
        self.assertEqual(set(self.faq.especiais), {"fora_de_escopo", "dados_da_conta", "pedido_humano"})

    def test_cabecalho_nao_vira_registro(self):
        self.assertFalse(any("FAQ interno" in r.texto for r in self.faq.registros))


class TestEstruturaInvalida(unittest.TestCase):
    CABECALHO = "Título\nDescrição. Última revisão: 01/2026.\n\n"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_sem_data_so_avisa(self):
        arq = _arquivo(self.pasta, "f.md", "Título\n\nPergunta?\nResposta.\n")
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            pares, revisao, _ = faq.ler_faq(arq, 1)
        self.assertIsNone(revisao)
        self.assertEqual(len(pares), 1)
        self.assertTrue(any("Última revisão" in str(a.message) for a in avisos))

    def test_numero_de_pares_errado_falha(self):
        arq = _arquivo(self.pasta, "f.md", self.CABECALHO + "P1?\nR1.\nP2?\nR2.\n")
        with self.assertRaisesRegex(faq.ErroFAQ, "2 pares encontrados; esperado 18"):
            faq.ler_faq(arq, 18)

    def test_resposta_faltando_falha(self):
        arq = _arquivo(self.pasta, "f.md", self.CABECALHO + "P1?\nR1.\nP2?\n")
        with self.assertRaisesRegex(faq.ErroFAQ, "número par"):
            faq.ler_faq(arq, 2)

    def test_pares_desalinhados_falham(self):
        arq = _arquivo(self.pasta, "f.md", self.CABECALHO + "P1?\nR1.\nR solta.\nP2?\n")
        with self.assertRaisesRegex(faq.ErroFAQ, "não começa com uma pergunta"):
            faq.ler_faq(arq, 2)

    def test_categoria_faltando_falha(self):
        cat = _arquivo(self.pasta, "c.toml", '[registros]\n1 = "a"\n')
        with self.assertRaisesRegex(faq.ErroFAQ, r"faltam \[2\]"):
            faq.ler_categorias(cat, 2)

    def test_categoria_igual_a_especial_falha(self):
        cat = _arquivo(self.pasta, "c.toml", '[registros]\n1 = "x"\n[especiais]\nx = "..."\n')
        with self.assertRaisesRegex(faq.ErroFAQ, "intenção especial"):
            faq.ler_categorias(cat, 1)


class TestPedacos(unittest.TestCase):
    def test_texto_curto_vira_um_pedaco(self):
        self.assertEqual(faq.dividir_em_pedacos("a b c", 5, 1), ["a b c"])

    def test_sobreposicao(self):
        texto = " ".join(str(i) for i in range(10))
        self.assertEqual(faq.dividir_em_pedacos(texto, 4, 1), ["0 1 2 3", "3 4 5 6", "6 7 8 9"])

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            faq.dividir_em_pedacos("a b", 3, 3)


if __name__ == "__main__":
    unittest.main()
