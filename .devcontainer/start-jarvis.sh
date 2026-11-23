#!/usr/bin/env bash
# Starts the Jarvis web dashboard when you open this codespace, so on a phone you only tap the link.
#
# Codespaces runs this each time you open (attach to) the codespace:
# - first time: starts Jarvis and prints the chat and dashboard links;
# - already running: prints the same links again instead of starting a second copy.
#
# Change it without editing this file, with Codespaces secrets (github.com/settings/codespaces):
#   JARVIS_AUTOSTART=0                 don't start Jarvis automatically
#   JARVIS_AUTOSTART_ARGS="--web ..."  start it with these options instead of the defaults below
set -u

DEFAULT_ARGS="--web --watch --auto --budget 5"
PORT=8765
LOG="${JARVIS_HOME:-$HOME/.jarvis}/web.log"

if [ "${JARVIS_AUTOSTART:-1}" = "0" ]; then
  echo "Jarvis auto-start is off (JARVIS_AUTOSTART=0). Start it yourself with: jarvis $DEFAULT_ARGS"
  exit 0
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  cat <<'EOF'
Jarvis needs your Anthropic API key before it can start.
  1. Create a key at https://platform.claude.com (API keys > Create key).
  2. Save it at https://github.com/settings/codespaces > New secret:
     name ANTHROPIC_API_KEY, repository access joshuawayzwright/jarvis.
  3. Reload this codespace (or stop and start it) so it picks the secret up.
EOF
  exit 0
fi

# Already running (the port answers): show the links from the running copy's output.
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "Jarvis is already running. Its links:"
  grep -E "https?://[^ ]*\?token=" "$LOG" 2>/dev/null | tail -n 2 | sed 's/\x1b\[[0-9;]*m//g' \
    || echo "  (not found in $LOG; restart the codespace to get new links)"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
read -r -a ARGS <<< "${JARVIS_AUTOSTART_ARGS:-$DEFAULT_ARGS}"
echo "Starting: jarvis ${ARGS[*]}"
echo "Tap the dashboard link below when it appears. Close this terminal or press Ctrl+C to stop Jarvis."
# Output goes to this terminal and to the log (for the "already running" case above), unbuffered so the
# links appear at once. exec makes Jarvis this process, so closing the terminal stops Jarvis too.
exec > >(tee "$LOG") 2>&1
PYTHONUNBUFFERED=1 exec jarvis "${ARGS[@]}"
