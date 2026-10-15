#!/usr/bin/env bash
# Moves cold files or directories to the NAS (/mnt/nas/thesis/archive, same path relative to $HOME)
# and leaves a symlink in their place, so readers keep working while the WSL disk stays small.
# Copies, checks every file by checksum, then deletes the local copy. Never use it on a path a
# running job is writing to.
#
#   bash pilot/scripts/nas_archive.sh PATH...
set -euo pipefail
ROOT=/mnt/nas/thesis/archive
ls /mnt/nas > /dev/null                     # triggers the systemd automount
mountpoint -q /mnt/nas || { echo "NAS is not mounted at /mnt/nas"; exit 1; }
bytes() { du -sb --apparent-size "$1" | cut -f1; }
for p in "$@"; do
  src=$(realpath -s "$p")
  [ -L "$src" ] && { echo "skip (already a link): $src"; continue; }
  [ -e "$src" ] || { echo "missing: $src"; exit 1; }
  dst=$ROOT/${src#"$HOME"/}
  mkdir -p "$(dirname "$dst")"
  rsync -rlt "$src" "$(dirname "$dst")/"
  # a checksum pass that would transfer nothing means every file arrived intact (the NAS keeps its own mtimes)
  [ -z "$(rsync -rlc --dry-run --itemize-changes "$src" "$(dirname "$dst")/")" ] || { echo "copy mismatch: $src"; exit 1; }
  size=$(bytes "$src")
  rm -rf "${src:?}"
  ln -s "$dst" "$src"
  echo "archived $src -> $dst ($size bytes)"
done
