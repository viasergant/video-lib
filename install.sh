#!/usr/bin/env bash
# install.sh — automated dependency installer for video-lib on Ubuntu 22.04 LTS
# Usage: sudo bash install.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FAILURES=()
APT_UPDATED=0

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; FAILURES+=("$1"); }

# ── step dispatcher ───────────────────────────────────────────────────────────
step() {
  local name="$1"
  echo ""
  echo "▶ $name"
  set +e
  "$name"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    fail "$name" "exited with code $rc"
  fi
}

# ── preflight ─────────────────────────────────────────────────────────────────
need_sudo() {
  if [[ $EUID -ne 0 ]]; then
    echo "This script requires sudo privileges. Re-run with: sudo bash install.sh"
    exit 1
  fi
}

check_os() {
  if command -v lsb_release &>/dev/null; then
    local id ver
    id=$(lsb_release -si 2>/dev/null || echo "Unknown")
    ver=$(lsb_release -sr 2>/dev/null || echo "0")
    if [[ "$id" != "Ubuntu" ]]; then
      warn "Detected OS: $id $ver — this script targets Ubuntu. Proceeding anyway."
    elif [[ "$ver" != "22.04" ]]; then
      warn "Detected Ubuntu $ver — tested against 22.04 LTS. Proceeding anyway."
    else
      ok "Ubuntu 22.04 LTS detected"
    fi
  else
    warn "lsb_release not found — cannot verify OS. Proceeding."
  fi
}

check_disk() {
  local avail_gb
  avail_gb=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {gsub("G",""); print $4}')
  if [[ "$avail_gb" -lt 5 ]]; then
    echo "Insufficient disk space: ${avail_gb}G available, 5G required (LLaVA model is ~4.1 GB)"
    exit 1
  fi
  ok "Disk space: ${avail_gb}G available"
}

# ── installation steps ────────────────────────────────────────────────────────
install_system_deps() {
  local pkgs=(python3 python3-pip python3-venv ffmpeg curl libgl1 libglib2.0-0)
  local missing=()
  for pkg in "${pkgs[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "ok installed"; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "All system packages already installed"
    return 0
  fi

  echo "  Installing: ${missing[*]}"
  if [[ $APT_UPDATED -eq 0 ]]; then
    apt-get update -qq
    APT_UPDATED=1
  fi
  apt-get install -y -qq "${missing[@]}"
  ok "System packages installed"

  # Validate Python version
  local pyver
  pyver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  local pymaj pymin
  pymaj=$(echo "$pyver" | cut -d. -f1)
  pymin=$(echo "$pyver" | cut -d. -f2)
  if [[ $pymaj -lt 3 ]] || [[ $pymaj -eq 3 && $pymin -lt 10 ]]; then
    echo "Python $pyver is below minimum 3.10. Install python3.10+ and re-run."
    return 1
  fi
  ok "Python $pyver satisfies >= 3.10"
}

install_ollama() {
  if command -v ollama &>/dev/null; then
    ok "Ollama already installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
    return 0
  fi

  echo "  Downloading Ollama installer..."
  if command -v curl &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
  elif command -v wget &>/dev/null; then
    wget -qO- https://ollama.ai/install.sh | sh
  else
    echo "curl or wget is required to download Ollama"
    return 1
  fi
  ok "Ollama installed"
}

start_ollama() {
  # Check if already running
  if curl -sf http://localhost:11434 &>/dev/null; then
    ok "Ollama daemon already running"
    return 0
  fi

  echo "  Starting Ollama daemon..."
  ollama serve &>/tmp/ollama-serve.log &
  local OLLAMA_PID=$!
  disown $OLLAMA_PID

  local attempts=0
  until curl -sf http://localhost:11434 &>/dev/null; do
    attempts=$((attempts + 1))
    if [[ $attempts -ge 30 ]]; then
      echo "Ollama daemon did not become ready after 30s. Check /tmp/ollama-serve.log"
      echo "Try running 'ollama serve' manually and check system resources."
      return 1
    fi
    sleep 1
  done
  ok "Ollama daemon is ready (${attempts}s)"
}

pull_llava() {
  if ollama list 2>/dev/null | grep -q "llava:latest"; then
    ok "llava:latest already present"
    return 0
  fi

  echo "  Pulling llava:latest — this may take several minutes (~4.1 GB)..."
  ollama pull llava:latest
  ok "llava:latest pulled"
}

create_venv() {
  if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    ok ".venv already exists"
    return 0
  fi

  echo "  Creating virtualenv at .venv..."
  if ! python3 -m venv "$SCRIPT_DIR/.venv" 2>/tmp/venv-create.log; then
    # python3-venv might be missing on minimal Ubuntu
    apt-get install -y -qq python3-venv
    python3 -m venv "$SCRIPT_DIR/.venv"
  fi
  ok "Virtualenv created at .venv"
}

install_python_deps() {
  local pip="$SCRIPT_DIR/.venv/bin/pip"
  echo "  Upgrading pip..."
  "$pip" install --upgrade pip -q
  echo "  Installing Python packages from requirements.txt..."
  "$pip" install -r "$SCRIPT_DIR/requirements.txt"
  ok "Python packages installed"
}

verify() {
  echo ""
  echo "── Post-install verification ──────────────────────────────────────────"
  local all_ok=0

  # python3 >= 3.10
  local pyver
  pyver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
  local pymaj pymin
  pymaj=$(echo "$pyver" | cut -d. -f1); pymin=$(echo "$pyver" | cut -d. -f2)
  if [[ $pymaj -ge 3 && $pymin -ge 10 ]]; then
    ok "python3 $pyver >= 3.10"
  else
    fail "python3" "version $pyver is below 3.10"; all_ok=1
  fi

  # ffmpeg
  if command -v ffmpeg &>/dev/null; then ok "ffmpeg on PATH"
  else fail "ffmpeg" "not found on PATH"; all_ok=1; fi

  # ollama
  if command -v ollama &>/dev/null; then ok "ollama on PATH"
  else fail "ollama" "not found on PATH"; all_ok=1; fi

  # Ollama daemon
  if curl -sf http://localhost:11434 &>/dev/null; then ok "Ollama daemon responding at :11434"
  else fail "ollama-daemon" "http://localhost:11434 not reachable — run 'ollama serve'"; all_ok=1; fi

  # llava:latest
  if ollama list 2>/dev/null | grep -q "llava:latest"; then ok "llava:latest present"
  else fail "llava:latest" "not in 'ollama list' — run 'ollama pull llava:latest'"; all_ok=1; fi

  # key Python imports
  if "$SCRIPT_DIR/.venv/bin/python" -c "import cv2, click" &>/dev/null; then ok "cv2, click importable in .venv"
  else fail "python-imports" "cv2 or click not importable — re-run pip install"; all_ok=1; fi

  # scan.py --help
  if "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/scan.py" --help &>/dev/null; then ok "scan.py --help exits 0"
  else fail "scan-entrypoint" "scan.py --help failed — check installation"; all_ok=1; fi

  return $all_ok
}

# ── main ──────────────────────────────────────────────────────────────────────
main() {
  echo "video-lib installer — Ubuntu 22.04 LTS"
  echo "======================================="
  need_sudo
  check_os
  check_disk

  for s in install_system_deps install_ollama start_ollama pull_llava create_venv install_python_deps verify; do
    step "$s"
  done

  echo ""
  if [[ ${#FAILURES[@]} -eq 0 ]]; then
    echo -e "${GREEN}All checks passed.${NC}"
    echo "Run: source .venv/bin/activate && scan --folder /path/to/videos"
    exit 0
  else
    echo -e "${RED}Installation incomplete. Failed steps: ${FAILURES[*]}${NC}"
    exit 1
  fi
}

main "$@"
