# rcfile for the interactive shell opened by `.\wsl.ps1 shell`.
# Loads the user's normal bashrc (errors from unrelated projects silenced),
# then the UAV project environment.
if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc" 2>/dev/null
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/setup_wsl_env.sh"
cd "$PROJECT_ROOT" || true
PS1="(uav) ${PS1:-\\u@\\h:\\w\\$ }"
