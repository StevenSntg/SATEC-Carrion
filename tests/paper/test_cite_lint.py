from paper.cite_lint import parse_citations, check_linearity

CUERPO_OK = "Texto [1], luego [2] y [3].\n## Referencias\n[1] A\n[2] B\n[3] C\n"
CUERPO_MAL = "Texto [1] y luego [17]. Más [3].\n## Referencias\n[1] A\n"

def test_parse_citations_orden_de_aparicion():
    assert parse_citations(CUERPO_OK) == [1, 2, 3]

def test_linearidad_correcta_no_reporta():
    assert check_linearity(CUERPO_OK) == []

def test_linearidad_detecta_salto():
    problemas = check_linearity(CUERPO_MAL)
    assert any("17" in p for p in problemas)
