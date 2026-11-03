#!/bin/sh
# Seed the persistent arie.yaml from the image default on first start, then run the API.
# Later starts keep the volume copy, so settings saved from the UI survive restarts.
set -eu

config="${ARIE_CONFIG:-/app/arie.yaml}"
if [ ! -f "$config" ]; then
    mkdir -p "$(dirname "$config")"
    cp /app/arie.yaml "$config"
    echo "arie: created $config from the image default"
fi
mkdir -p /app/data/papers

exec "$@"
