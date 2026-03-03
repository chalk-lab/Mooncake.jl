#!/bin/bash
# SessionStart hook: install Julia for Claude Code on the web.
# Skips entirely when running locally (CLI).

set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

JULIA_VERSION="1.12.5"
JULIA_INSTALL_DIR="/usr/local"

# Skip if the correct version is already installed.
if command -v julia &>/dev/null; then
    INSTALLED="$(julia --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+')"
    if [ "$INSTALLED" = "$JULIA_VERSION" ]; then
        echo "Julia $JULIA_VERSION already installed – skipping."
        exit 0
    fi
fi

MAJOR_MINOR="${JULIA_VERSION%.*}"
TARBALL_URL="https://julialang-s3.julialang.org/bin/linux/x64/${MAJOR_MINOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz"

echo "Installing Julia $JULIA_VERSION …"
curl -fsSL "$TARBALL_URL" | tar xz -C "$JULIA_INSTALL_DIR" --strip-components=1

# Persist PATH for subsequent Bash commands in this session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    echo "export PATH=\"${JULIA_INSTALL_DIR}/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

julia --version
echo "Julia $JULIA_VERSION ready."
