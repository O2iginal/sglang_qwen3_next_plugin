#!/usr/bin/env bash
set -euo pipefail

CHAT_HOST=${CHAT_HOST:-127.0.0.1}
CHAT_PORT=${CHAT_PORT:-30110}
CHAT_MODEL=${CHAT_MODEL:-default}
CHAT_PROMPT=${CHAT_PROMPT:-简短介绍一下你自己，用中文回答}
CHAT_TEMPERATURE=${CHAT_TEMPERATURE:-0.7}
CHAT_MAX_TOKENS=${CHAT_MAX_TOKENS:-128}

payload=$(cat <<EOF
{
  "model": "${CHAT_MODEL}",
  "messages": [
    {
      "role": "user",
      "content": "${CHAT_PROMPT}"
    }
  ],
  "temperature": ${CHAT_TEMPERATURE},
  "max_tokens": ${CHAT_MAX_TOKENS},
  "stream": false
}
EOF
)

curl "http://${CHAT_HOST}:${CHAT_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}"
