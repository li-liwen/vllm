#!/usr/bin/env bash
# DSv4-on-gfx908 entrypoint.
#
# Behavior:
#   * No args, or "--help"          -> show vllm CLI help.
#   * First arg starts with "-"     -> treat as vllm CLI args.
#   * First arg is "serve" / "bench"/ "run" -> forward verbatim to vllm.
#   * Anything else                 -> exec the command as-is (so the
#                                      container is still useful for
#                                      ad-hoc shells, e.g.
#                                      `docker run ... bash`).

set -euo pipefail

if [[ $# -eq 0 ]]; then
    exec vllm --help
fi

case "$1" in
    -h|--help)
        exec vllm --help
        ;;
    -*)
        exec vllm "$@"
        ;;
    serve|bench|run|chat|complete)
        exec vllm "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
