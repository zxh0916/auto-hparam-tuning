#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_SRC="$SCRIPT_DIR/skills"

if [ $# -ge 1 ]; then
    SKILLS_DST="$(realpath "$1")/.claude/skills"
else
    SKILLS_DST="${HOME}/.claude/skills"
fi

mkdir -p "$SKILLS_DST"

for skill_dir in "$SKILLS_SRC"/*/; do
    skill_name="$(basename "$skill_dir")"
    target="$SKILLS_DST/$skill_name"

    if [ -L "$target" ]; then
        echo "skip (already linked): $skill_name"
    elif [ -e "$target" ]; then
        echo "skip (already exists, not a symlink): $skill_name"
    else
        ln -s "$skill_dir" "$target"
        echo "linked: $skill_name -> $target"
    fi
done
