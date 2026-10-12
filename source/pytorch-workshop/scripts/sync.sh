#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'usage: sync.sh [-h]

Sync the local workspace to remote.
'
    exit
fi

cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

main() {
    workspace_exclude_patterns=(
        ".cache"
        ".venv"
        ".pytest_cache"
        ".vscode"
        "__pycache__"
        "/data/"
        "/libs/"
        "/models/"
        "/logs/"
        "/wandb/"
        "*.db"
    )

    # Read remotes from configuration file
    config_file="$(dirname "$0")/configs/sync.conf"
    if [ ! -f "$config_file" ]; then
        echo "[error] Configuration file not found at $config_file"
        exit 1
    fi

    # Sync local workspace to each remote
    workspace_exclude_opts=()
    for pattern in "${workspace_exclude_patterns[@]}"; do
        workspace_exclude_opts+=("--exclude" "$pattern")
    done
    while IFS= read -r remote || [ -n "$remote" ]; do
        # Skip empty lines and comments
        [[ -z "$remote" || "$remote" =~ ^[[:space:]]*# ]] && continue
        echo "[info] Syncing $(pwd) to $remote..."
        rsync -azhv "${workspace_exclude_opts[@]}" . "$remote"
    done < "$config_file"

}

main "$@"
