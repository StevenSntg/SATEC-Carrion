"""Verifica la linealidad de citas IEEE/ACM: primera aparición en orden 1..N."""
import re
import sys

_CITE = re.compile(r"\[(\d+)\]")


def _split_body(md_text):
    m = re.search(r"^##\s+Referencias", md_text, flags=re.MULTILINE)
    return md_text[:m.start()] if m else md_text


def parse_citations(md_text):
    body = _split_body(md_text)
    vistos, orden = set(), []
    for n in (int(x) for x in _CITE.findall(body)):
        if n not in vistos:
            vistos.add(n); orden.append(n)
    return orden


def check_linearity(md_text):
    orden = parse_citations(md_text)
    problemas = []
    for esperado, real in enumerate(orden, start=1):
        if real != esperado:
            problemas.append(
                f"Cita [{real}] aparece donde se esperaba [{esperado}] "
                f"(orden de primera aparición roto).")
            break
    return problemas


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/SATEC_articulo_ACM.md"
    with open(path, encoding="utf-8") as f:
        problemas = check_linearity(f.read())
    if problemas:
        print("\n".join(problemas)); sys.exit(1)
    print("[OK] Citas en orden lineal de aparición.")


if __name__ == "__main__":
    main()
