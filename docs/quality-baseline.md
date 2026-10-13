# Baseline de calidad

Estado de referencia antes del saneamiento (rama `dev`).

## Entorno

- `python -m venv .venv` (con `--system-site-packages`, ya que el host provee
  `torch`, `ultralytics`, `opencv-python`, `flask`; así se evitan descargas de
  varios GB de ruedas CUDA).
- `.venv/bin/pip install -e ".[dev]"` + `vulture bandit radon`.

## Resultados

| Herramienta | Resultado |
| --- | --- |
| `pytest -q` | 90 passed |
| Cobertura total | 78% (`common` 88%, `web_app` 52%) |
| `ruff check .` | OK |
| `ruff format --check .` | 22 files already formatted |
| `mypy` | Success: no issues found in 15 source files |

### Cobertura por módulo

| Módulo | Cover |
| --- | --- |
| `common/cli.py` | 100% |
| `common/config.py` | 100% |
| `common/detector.py` | 90% |
| `common/frames.py` | 88% |
| `common/geometry.py` | 83% |
| `common/isapi.py` | 96% |
| `common/lanes.py` | 59% |
| `common/rtsp.py` | 100% |
| `common/utils.py` | 98% |
| `web_app/app.py` | 52% |

## Hallazgos

### vulture (`--min-confidence 70`)

Solo aparecen variables/parámetros intencionalmente sin usar en los stubs de
`tests/test_common_units.py` (p.ej. `chunk_size`, `*a, **k`, `prop`,
`source`, `classes`, `b, c`). Son firmas de compatibilidad; se resolverán
renombrándolas con prefijo `_` para que vulture las ignore.

### bandit (`-q -r common web_app`)

3 hallazgos LOW:

- `common/detector.py:111` — B112 `try/except/continue`.
- `common/frames.py:273` — B112 `try/except/continue`.
- `web_app/app.py:277` — B110 `try/except/pass`.

### radon cc

- `web_app/app.py DetectorCamera._loop` — C (11), candidato a dividir.
- `common/lanes.py LaneDetector.draw_lanes` — B (7).
- `common/isapi.py save_isapi_snapshot` — B (6).

## Objetivos del saneamiento

1. Reducir duplicación entre `capture_preroll.py` y
   `capture_preroll_lanes.py` (núcleo pre-roll común).
2. Modularizar `common/frames.py` y `web_app/app.py`.
3. Optimizar recomputos/copias sin cambiar comportamiento.
4. Ampliar cobertura (`common/lanes.py`, `common/isapi.py`, pre-roll).
5. Resolver hallazgos razonables de bandit/vulture.
