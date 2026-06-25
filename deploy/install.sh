#!/usr/bin/env bash
# Install palazzo as a systemd service. Idempotent.
# Usage:  sudo ./install.sh /path/to/palazzo
# Places the binary at /opt/palazzo/bin/palazzo, installs the unit file,
# creates a system user, and reloads systemd. Does NOT start the service —
# review /etc/palazzo/env first, then:
#   sudo systemctl enable --now palazzo

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "run as root" >&2
    exit 1
fi

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
    echo "usage: sudo $0 <path-to-palazzo-binary>" >&2
    exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

# System user & dirs
if ! id palazzo >/dev/null 2>&1; then
    useradd --system --shell /usr/sbin/nologin --home-dir /var/lib/palazzo --create-home palazzo
fi
install -d -o palazzo -g palazzo -m 0750 /var/lib/palazzo
install -d -m 0755 /opt/palazzo /opt/palazzo/bin /etc/palazzo

# Binary — back up the current one (named by its version) before overwriting,
# then keep only the 2 most recent backups so /opt/palazzo/bin doesn't grow
# unbounded (each binary is ~35 MB; this used to pile up across every deploy).
if [[ -x /opt/palazzo/bin/palazzo ]]; then
    cur="$(/opt/palazzo/bin/palazzo --version 2>/dev/null | awk '{print $2}')"
    cp /opt/palazzo/bin/palazzo "/opt/palazzo/bin/palazzo.bak-${cur:-unknown}"
fi
install -m 0755 "$BIN" /opt/palazzo/bin/palazzo
# Prune: keep the 2 newest backups (by mtime), drop the rest. Matches both the
# old `palazzo.bak-pre-vX` and new `palazzo.bak-VERSION` names.
ls -1t /opt/palazzo/bin/palazzo.bak-* 2>/dev/null | tail -n +3 | xargs -r rm -f

# Env file: copy example if /etc/palazzo/env doesn't exist; never overwrite.
if [[ ! -f /etc/palazzo/env ]]; then
    install -m 0640 -o root -g palazzo "$HERE/env.example" /etc/palazzo/env
    echo "wrote default /etc/palazzo/env — review before starting"
fi

# systemd unit
install -m 0644 "$HERE/palazzo.service" /etc/systemd/system/palazzo.service
systemctl daemon-reload

cat <<'EOF'

palazzo installed. Next steps:
  1. Edit /etc/palazzo/env to taste.
  2. sudo systemctl enable --now palazzo
  3. sudo systemctl status palazzo
  4. curl -sS -X POST http://localhost:6334/mcp ... (see README for full MCP handshake)
EOF
