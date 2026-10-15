#!/usr/bin/env bash
# Instalador idempotente del core (accelerated_drug_design_core) para Ubuntu.
# Revisa qué hay en el servidor y solo hace lo que falta o cambió.
#
# Uso:
#   ./install_core.sh --check     # solo reporta qué haría, no cambia nada
#   ./install_core.sh             # instala / actualiza lo necesario
#   ./install_core.sh --rebuild   # fuerza reconstruir la imagen
#
# Variables opcionales: REPO_URL, REPO_DIR, IMAGE, CONTAINER, PORT
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/A01795943/accelerated_drug_design_core.git}"
REPO_DIR="${REPO_DIR:-$HOME/accelerated_drug_design_core}"
IMAGE="${IMAGE:-drug-accelerator}"
CONTAINER="${CONTAINER:-core}"
PORT="${PORT:-8000}"
CUDA_TEST_IMAGE="${CUDA_TEST_IMAGE:-nvidia/cuda:12.6.0-base-ubuntu22.04}"

CHECK=0
REBUILD=0
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --rebuild) REBUILD=1 ;;
    *) echo "Opción desconocida: $a (usa --check o --rebuild)"; exit 1 ;;
  esac
done

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

have() { command -v "$1" >/dev/null 2>&1; }
ok()   { echo "[OK]      $*"; }
todo() { echo "[HACER]   $*"; }
warn() { echo "[AVISO]   $*" >&2; }
info() { echo "[...]     $*"; }

NO_GPU=0
DRIVER_OK=1
PENDING=0   # cuenta de cosas por hacer (modo --check)

mark_todo() { todo "$*"; PENDING=$((PENDING + 1)); }

# ---------------------------------------------------------------- sistema
if [ -r /etc/os-release ]; then . /etc/os-release; fi
if [ "${ID:-}" != "ubuntu" ]; then
  echo "Este script solo soporta Ubuntu (detectado: ${ID:-desconocido})."; exit 1
fi

# ---------------------------------------------------------------- base
step_base() {
  local pkgs=() pair cmd pkg
  for pair in git:git curl:curl lspci:pciutils; do
    cmd="${pair%%:*}"; pkg="${pair##*:}"
    have "$cmd" || pkgs+=("$pkg")
  done
  if [ ${#pkgs[@]} -eq 0 ]; then ok "git, curl y pciutils instalados"; return; fi
  if [ $CHECK -eq 1 ]; then mark_todo "Instalar paquetes base: ${pkgs[*]}"; return; fi
  info "Instalando paquetes base: ${pkgs[*]}"
  $SUDO apt-get update -y
  $SUDO apt-get install -y ca-certificates "${pkgs[@]}"
}

# ---------------------------------------------------------------- docker
install_docker() {
  $SUDO apt-get update -y
  $SUDO apt-get install -y ca-certificates curl
  $SUDO install -m 0755 -d /etc/apt/keyrings
  $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  $SUDO chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  $SUDO apt-get update -y
  $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

step_docker() {
  if ! have docker; then
    if [ $CHECK -eq 1 ]; then mark_todo "Instalar Docker"; return; fi
    info "Instalando Docker"
    install_docker
  else
    ok "Docker instalado ($(docker --version))"
  fi
  if have docker && ! $SUDO docker info >/dev/null 2>&1; then
    if [ $CHECK -eq 1 ]; then mark_todo "Iniciar el servicio docker"; return; fi
    info "Iniciando servicio docker"
    $SUDO systemctl enable --now docker
  fi
}

# ---------------------------------------------------------------- driver NVIDIA
step_driver() {
  if ! lspci 2>/dev/null | grep -qi nvidia; then
    warn "No se detectó GPU NVIDIA: se omiten driver y toolkit (el core correrá sin GPU)."
    NO_GPU=1
    return
  fi
  if nvidia-smi >/dev/null 2>&1; then
    ok "Driver NVIDIA funcionando ($(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1))"
    return
  fi
  DRIVER_OK=0
  if [ $CHECK -eq 1 ]; then
    mark_todo "Instalar/reparar driver NVIDIA (nvidia-smi no funciona; kernel actual: $(uname -r))"
    return
  fi

  info "nvidia-smi no funciona; instalando/reparando driver"
  $SUDO apt-get update -y
  $SUDO apt-get install -y ubuntu-drivers-common
  if ! dpkg -l | grep -q '^ii  nvidia-driver-'; then
    $SUDO ubuntu-drivers install
  fi

  # Módulos precompilados y firmados (sirven con Secure Boot) para el kernel EN USO.
  local ver pkg
  ver="$(dpkg -l | awk '/^ii  nvidia-driver-[0-9]+/ {print $2}' | head -1 | sed -E 's/^nvidia-driver-([0-9]+).*/\1/')"
  if [ -n "$ver" ]; then
    for pkg in "linux-modules-nvidia-${ver}-open-$(uname -r)" "linux-modules-nvidia-${ver}-$(uname -r)"; do
      if apt-cache policy "$pkg" 2>/dev/null | grep -q 'Candidate: [^(]'; then
        info "Instalando módulos para el kernel actual: $pkg"
        $SUDO apt-get install -y "$pkg"
        break
      fi
    done
  fi

  $SUDO modprobe nvidia 2>/dev/null || true
  if nvidia-smi >/dev/null 2>&1; then
    DRIVER_OK=1
    ok "Driver NVIDIA funcionando"
  else
    warn "Driver instalado pero aún no carga. Reinicia el servidor y vuelve a ejecutar este script."
    exit 2
  fi
}

# ---------------------------------------------------------------- nvidia container toolkit
step_toolkit() {
  if [ $NO_GPU -eq 1 ]; then return; fi
  if [ $DRIVER_OK -eq 0 ]; then warn "Toolkit omitido: primero debe funcionar el driver."; return; fi
  if ! have docker; then
    [ $CHECK -eq 1 ] && mark_todo "Instalar nvidia-container-toolkit (requiere Docker)"
    return
  fi

  if ! have nvidia-ctk; then
    if [ $CHECK -eq 1 ]; then mark_todo "Instalar nvidia-container-toolkit"; return; fi
    info "Instalando nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update -y
    $SUDO apt-get install -y nvidia-container-toolkit
  else
    ok "nvidia-container-toolkit instalado"
  fi

  if $SUDO docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
    ok "Docker tiene el runtime nvidia configurado"
  else
    if [ $CHECK -eq 1 ]; then mark_todo "Configurar runtime nvidia en Docker"; return; fi
    info "Configurando runtime nvidia en Docker"
    $SUDO nvidia-ctk runtime configure --runtime=docker
    $SUDO systemctl restart docker
  fi

  if [ $CHECK -eq 0 ]; then
    if $SUDO docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi >/dev/null 2>&1; then
      ok "Los contenedores ven la GPU"
    else
      warn "Los contenedores NO ven la GPU; revisa la salida de: sudo docker run --rm --gpus all $CUDA_TEST_IMAGE nvidia-smi"
      exit 3
    fi
  fi
}

# ---------------------------------------------------------------- repositorio
step_repo() {
  if ! have git; then
    [ $CHECK -eq 1 ] && mark_todo "Clonar $REPO_URL (requiere git)"
    return
  fi
  if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch -q || warn "No se pudo hacer fetch (¿sin red?)"
    local_head="$(git -C "$REPO_DIR" rev-parse HEAD)"
    remote_head="$(git -C "$REPO_DIR" rev-parse '@{u}' 2>/dev/null || echo "$local_head")"
    if [ "$local_head" = "$remote_head" ]; then
      ok "Repo al día en $REPO_DIR (${local_head:0:7})"
    elif [ $CHECK -eq 1 ]; then
      mark_todo "Actualizar repo (git pull)"
    else
      info "Actualizando repo"
      git -C "$REPO_DIR" pull --ff-only || warn "git pull falló (cambios locales?); se usa la versión actual."
    fi
  else
    if [ $CHECK -eq 1 ]; then mark_todo "Clonar $REPO_URL en $REPO_DIR"; return; fi
    info "Clonando repo"
    git clone "$REPO_URL" "$REPO_DIR"
  fi
}

# ---------------------------------------------------------------- imagen
step_build() {
  if ! have docker || [ ! -d "$REPO_DIR/.git" ]; then
    [ $CHECK -eq 1 ] && mark_todo "Construir imagen $IMAGE (faltan Docker o repo)"
    return
  fi
  if [ ! -f "$REPO_DIR/docker/Dockerfile" ]; then
    warn "No existe $REPO_DIR/docker/Dockerfile; ajusta la ruta en este script."
    exit 4
  fi
  local commit img_commit
  commit="$(git -C "$REPO_DIR" rev-parse HEAD)"
  img_commit="$($SUDO docker image inspect "$IMAGE" --format '{{ index .Config.Labels "commit" }}' 2>/dev/null || true)"
  if [ $REBUILD -eq 0 ] && [ "$img_commit" = "$commit" ]; then
    ok "Imagen $IMAGE al día (${commit:0:7})"
    return
  fi
  if [ $CHECK -eq 1 ]; then mark_todo "Construir imagen $IMAGE (código nuevo o imagen inexistente)"; return; fi
  info "Construyendo imagen $IMAGE (puede tardar bastante)"
  ( cd "$REPO_DIR" && $SUDO docker build -f docker/Dockerfile --label "commit=$commit" -t "$IMAGE" . )
}

# ---------------------------------------------------------------- contenedor
step_run() {
  if ! have docker || ! $SUDO docker image inspect "$IMAGE" >/dev/null 2>&1; then
    [ $CHECK -eq 1 ] && mark_todo "Levantar contenedor $CONTAINER (falta la imagen)"
    return
  fi
  local img_id cur_img running
  img_id="$($SUDO docker image inspect "$IMAGE" --format '{{.Id}}')"
  cur_img="$($SUDO docker inspect "$CONTAINER" --format '{{.Image}}' 2>/dev/null || true)"
  running="$($SUDO docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null || echo false)"

  if [ "$cur_img" = "$img_id" ] && [ "$running" = "true" ]; then
    ok "Contenedor $CONTAINER corriendo con la imagen actual"
    return
  fi
  if [ $CHECK -eq 1 ]; then mark_todo "Crear/reiniciar contenedor $CONTAINER"; return; fi

  mkdir -p "$REPO_DIR/outputs" "$REPO_DIR/logs"
  if $SUDO docker inspect "$CONTAINER" >/dev/null 2>&1; then
    info "Reemplazando contenedor $CONTAINER"
    $SUDO docker rm -f "$CONTAINER" >/dev/null
  fi
  local gpu_args=()
  if [ $NO_GPU -eq 0 ]; then gpu_args=(--gpus all); fi
  info "Levantando contenedor $CONTAINER"
  $SUDO docker run -d --name "$CONTAINER" --restart unless-stopped "${gpu_args[@]}" \
    --shm-size=8g -p "$PORT:8000" \
    -v "$REPO_DIR/outputs:/workspace/outputs" \
    -v "$REPO_DIR/logs:/workspace/logs" \
    "$IMAGE" >/dev/null
}

# ---------------------------------------------------------------- health
step_health() {
  if ! have curl; then return; fi
  if [ $CHECK -eq 1 ]; then
    if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then ok "API responde en :$PORT/health"; fi
    return
  fi
  info "Esperando a la API en :$PORT/health"
  local i
  for i in $(seq 1 30); do
    if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
      ok "API arriba: $(curl -fsS "http://localhost:$PORT/health")"
      return
    fi
    sleep 5
  done
  warn "La API no respondió en 150 s. Revisa: sudo docker logs $CONTAINER"
  exit 5
}

step_base
step_docker
step_driver
step_toolkit
step_repo
step_build
step_run
step_health

echo
if [ $CHECK -eq 1 ]; then
  if [ $PENDING -eq 0 ]; then echo "Todo en orden: no hay nada que cambiar."; else echo "Pendientes: $PENDING. Ejecuta sin --check para aplicarlos."; fi
else
  echo "Listo."
fi
