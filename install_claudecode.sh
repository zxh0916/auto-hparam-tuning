#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_SRC="$SCRIPT_DIR/skills"
AGENTS_SRC="$SCRIPT_DIR/agents"

if [ $# -ge 1 ]; then
    CLAUDE_DST="$(realpath "$1")/.claude"
else
    CLAUDE_DST="${HOME}/.claude"
fi

SKILLS_DST="$CLAUDE_DST/skills"
AGENTS_DST="$CLAUDE_DST/agents"

link_entry() {
    local src="$1" dst_dir="$2"
    local name
    name="$(basename "$src")"
    local target="$dst_dir/$name"

    if [ -L "$target" ]; then
        echo "skip (already linked): $name"
    elif [ -e "$target" ]; then
        echo "skip (already exists, not a symlink): $name"
    else
        ln -s "$src" "$target"
        echo "linked: $name -> $target"
    fi
}

mkdir -p "$SKILLS_DST"
for skill_dir in "$SKILLS_SRC"/*/; do
    link_entry "$skill_dir" "$SKILLS_DST"
done

if [ -d "$AGENTS_SRC" ]; then
    mkdir -p "$AGENTS_DST"
    for agent_md in "$AGENTS_SRC"/*.md; do
        [ -e "$agent_md" ] || continue
        link_entry "$agent_md" "$AGENTS_DST"
    done
fi

# Add permission rule to settings file
SCRIPT_PATH="$SKILLS_DST/auto-hparam-tuning/scripts/session_manager.py"
PERMISSION="Bash(python $SCRIPT_PATH:*)"

if [ $# -ge 1 ]; then
    SETTINGS_FILE="$CLAUDE_DST/settings.local.json"
else
    SETTINGS_FILE="$CLAUDE_DST/settings.json"
fi

python3 - "$SETTINGS_FILE" "$PERMISSION" <<'EOF'
import sys, json, os

settings_file, permission = sys.argv[1], sys.argv[2]

if os.path.exists(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
else:
    settings = {}

perms = settings.setdefault("permissions", {})
allow = perms.setdefault("allow", [])

if permission in allow:
    print(f"skip (already present): {permission}")
else:
    allow.append(permission)
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    print(f"added permission: {permission}")
    print(f"  -> session_manager.py can now run without prompting in Claude Code")
EOF
