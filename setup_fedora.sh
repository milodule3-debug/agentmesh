#!/usr/bin/env bash
# AgentMesh — Fedora setup
set -e
echo "── AgentMesh Fedora Setup ──────────────────────"

python3 --version

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q
pip install requests python-dotenv honcho-ai

if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Edit .env and add your API keys"
fi

echo ""
echo "  Activate env:  source .venv/bin/activate"
echo "  Test keys:     python test_providers.py"
echo "  Run:           python run.py \"your goal\""
echo "── Done ────────────────────────────────────────"
