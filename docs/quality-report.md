# Reporte final de saneamiento

Cierre del saneamiento de calidad sobre la rama `dev` (partiendo de
`docs/quality-baseline.md`).

## Estado final de las herramientas

| Herramienta | Baseline | Final |
| --- | --- | --- |
| `pytest -q` | 90 passed | 121 passed |
| Cobertura total | 78% | 85% |
| `ruff check .` | OK | OK |
| `ruff format --check .` | OK | OK |
| `mypy` | OK (15 files) | OK (22 files) |
| `bandit -q -r common web_app` | 3 LOW | 0 |
| `vulture --min-confidence 70` | 23 avisos (stubs) | 0 |

### Cobertura por módulo (final)

| Módulo | Cover |
| --- | --- |
| `common/cli.py` | 100% |
| `common/config.py` | 100% |
| `common/frames.py` | 100% (fachada) |
| `common/isapi.py` | 100% |
| `common/lanes.py` | 100% |
| `common/preroll.py` | 100% |
| `common/rtsp.py` | 100% |
| `common/sharpness.py` | 100% |
| `common/utils.py` | 98% |
| `common/detector.py` | 89% |
| `common/grabbers.py` | 88% |
| `common/geometry.py` | 83% |
| `web_app/config.py` | 100% |
| `web_app/security.py` | 100% |
| `web_app/storage.py` | 100% |
| `web_app/app.py` | 67% |
| `web_app/camera.py` | 24% |

Los bucles de video (`web_app/camera.py`, `common/grabbers.py`) y ramas de
dibujo (`common/geometry.py`) son difíciles de cubrir sin captura real; el
gate de CI (`pytest --cov-fail-under=70`) queda holgadamente superado.

## Commits del saneamiento

1. `docs(quality)` — baseline verificable.
2. `refactor(preroll)` — núcleo común `common/preroll.py` (dedupe de los dos
   scripts de pre-roll).
3. `refactor(frames)` — `common/grabbers.py` (base común) + `common/sharpness.py`;
   `common/frames.py` queda como fachada.
4. `refactor(web)` — `web_app/{config,security,camera,storage}.py`.
5. `perf` — copias/recomputos evitados.
6. `test(lanes)` — `common/lanes.py` a 100%.
7. `test(isapi)` — timeouts/errores y reenvío de timeout.
8. `test(web)` — config/security/storage a 100%.
9. `test(preroll)` — settings, road filter, guardado y scheduler.
10. `hardening` — bandit/vulture a cero.
11. `refactor(grabbers)` — extracción que baja `_capture_loop` de C(17) a C(14).

## Hallazgos corregidos

- **Duplicación pre-roll**: `capture_preroll.py` y `capture_preroll_lanes.py`
  comparten ajustes, filtro de carretera, guardado y scheduler; cada script
  solo orquesta. Console-scripts intactos.
- **`common/frames.py` (370 L)**: dividido en `grabbers` + `sharpness` con una
  base común que elimina ~90 líneas duplicadas; API histórica preservada.
- **`web_app/app.py` (369 L → 192 L)**: rutas y estado globl separados de
  configuración, seguridad, cámara y almacenamiento.
- **bandit**: B110 (except/pass) y B112 (except/continue) resueltos con
  logging; 0 hallazgos.
- **vulture**: parámetros intencionalmente no usados de los stubs de test
  renombrados con prefijo `_`; 0 hallazgos.
- **optimización**: se elimina una copia de frame por iteración en el bucle
  web y se pre-renderiza el frame de "reconectando"; `_separate_lines` hace un
  solo `reshape`.

## Sin cambio de comportamiento

Todos los refactors conservan el comportamiento observable (mismas firmas
públicas, mismos console-scripts, mismos mensajes de estado); la suite
`pytest` se mantiene verde en cada commit.
