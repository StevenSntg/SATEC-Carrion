"""Constantes y parámetros por defecto del pipeline SATEC."""

WEEKS = list(range(1, 53))  # semanas epidemiológicas 1..52

DEFAULT_PARAMS = {
    "horizon": 4,        # semanas hacia adelante para el target
    "n_ref": 5,          # años de referencia para el canal endémico
    "min_ref": 3,        # mínimo de años de referencia exigidos
    "min_cases": 2,      # casos mínimos para considerar brote
    "endemic_min_casos": 10,   # casos históricos mínimos para provincia endémica
    "endemic_min_anios": 3,    # años distintos con casos para provincia endémica
}

RAW_COLUMNS = [
    "departamento", "provincia", "distrito", "localidad", "enfermedad",
    "ano", "semana", "diagnostic", "diresa", "ubigeo", "localcod",
    "edad", "tipo_edad", "sexo",
]
