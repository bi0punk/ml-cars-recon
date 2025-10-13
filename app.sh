#!/usr/bin/env bash
# Arregla definitivamente errores tipo:
# OSError: [Errno 28] No queda espacio en el dispositivo (instalando torch, etc.)
# Mueve el directorio temporal a $HOME/tmp y ajusta pip/conda.
# Uso: source fix_tmp_for_pip_conda.sh

set -euo pipefail

# Detectar archivos de configuración de shell
RC_FILES=()
[[ -f "$HOME/.bashrc" ]] && RC_FILES+=("$HOME/.bashrc")
[[ -f "$HOME/.zshrc"  ]] && RC_FILES+=("$HOME/.zshrc")
# Si no existe ninguno, usamos .bashrc por defecto
if [[ ${#RC_FILES[@]} -eq 0 ]]; then
  RC_FILES+=("$HOME/.bashrc")
  touch "$HOME/.bashrc"
fi

# 1) Crear directorio temporal en $HOME (amplio) y exportar variables
USER_TMP="$HOME/tmp"
mkdir -p "$USER_TMP"

# Permisos: privado para el usuario (suficiente para TMPDIR)
chmod 700 "$USER_TMP"

# Variables a exportar (aplican a muchas herramientas, incluido pip/conda)
EXPORT_LINES=$'# >>> FIX TMP for pip/conda >>>\nexport TMPDIR="$HOME/tmp"\nexport TEMP="$HOME/tmp"\nexport TMP="$HOME/tmp"\n# Evitar caché pesada de pip por defecto (opcional, ahorra espacio)\nexport PIP_NO_CACHE_DIR=1\n# <<< FIX TMP for pip/conda <<<\n'

for rc in "${RC_FILES[@]}"; do
  if ! grep -q 'FIX TMP for pip/conda' "$rc" 2>/dev/null; then
    printf "%s\n" "$EXPORT_LINES" >> "$rc"
    echo "[OK] Añadido bloque TMPDIR a $rc"
  else
    echo "[OK] Bloque TMPDIR ya presente en $rc"
  fi
done

# Exportar para la sesión actual (porque estamos haciendo 'source')
export TMPDIR="$USER_TMP"
export TEMP="$USER_TMP"
export TMP="$USER_TMP"
export PIP_NO_CACHE_DIR=1

# 2) Configurar pip para usar la caché en $HOME (por si necesitas caché)
#    Nota: Con PIP_NO_CACHE_DIR=1 pip no la usará; si luego quieres caché,
#    puedes hacer: pip config set global.no-cache-dir false
mkdir -p "$HOME/.cache/pip"
pip_cmds=()
command -v pip  >/dev/null 2>&1 && pip_cmds+=("pip")
command -v pip3 >/dev/null 2>&1 && pip_cmds+=("pip3")
for pcmd in "${pip_cmds[@]}"; do
  if "$pcmd" --version >/dev/null 2>&1; then
    "$pcmd" config set global.cache-dir "$HOME/.cache/pip" >/dev/null || true
    "$pcmd" config set global.timeout 120 >/dev/null || true
    # Dejar explícito no-cache-dir=true para minimizar espacio
    "$pcmd" config set global.no-cache-dir true >/dev/null || true
    echo "[OK] Configurado $pcmd (cache-dir en \$HOME y no-cache-dir=true)"
  fi
done

# 3) Configurar conda para usar directorios en $HOME (pkgs/envs) y respetar TMPDIR
if command -v conda >/dev/null 2>&1; then
  # Asegurar rutas locales
  mkdir -p "$HOME/.conda/pkgs" "$HOME/.conda/envs"
  conda config --add pkgs_dirs "$HOME/.conda/pkgs"  >/dev/null 2>&1 || true
  conda config --add envs_dirs "$HOME/.conda/envs"  >/dev/null 2>&1 || true
  # Evitar escribir en /tmp durante extracciones grandes (usa TMPDIR)
  export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
  echo "[OK] Configurado conda (pkgs_dirs/envs_dirs en \$HOME)"
else
  echo "[INFO] conda no está en PATH. Saltando configuración de conda."
fi

# 4) Crear alias útiles para instalaciones grandes
add_alias() {
  local name="$1"; shift
  local cmd="$*"
  for rc in "${RC_FILES[@]}"; do
    if ! grep -q "alias $name=" "$rc" 2>/dev/null; then
      echo "alias $name='$cmd'" >> "$rc"
      echo "[OK] Alias '$name' añadido a $rc"
    fi
  done
}

# pip sin caché explícita
add_alias pipw "TMPDIR=\"$USER_TMP\" PIP_NO_CACHE_DIR=1 pip"
# conda con TMPDIR explícito
if command -v conda >/dev/null 2>&1; then
  add_alias condaw "TMPDIR=\"$USER_TMP\" conda"
fi

# 5) Mostrar estado final
echo "------------------------------------------------------------"
echo "[INFO] TMPDIR actual: $TMPDIR"
echo "[INFO] Verificación de espacio en /tmp y \$TMPDIR:"
df -h /tmp "$TMPDIR" || true
echo "------------------------------------------------------------"
echo "[OK] Listo. Cierra y abre la terminal o ejecuta:"
echo "     source ~/.bashrc    # o  source ~/.zshrc"
echo "Para instalar con pip usando este fix, puedes usar el alias:"
echo "     pipw install paquete_grande"
echo "Con conda:"
echo "     condaw install paquete_grande"
