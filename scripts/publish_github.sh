#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

repo_name="${1:-gptpatch-reversible-cl}"
owner="${2:-amazedsaint}"
description="Reversible continual-learning patch for GPT-2 (invertible sidecars + replay/distill/rollback)"

echo "[gh] refreshing auth scopes (repo, read:org, workflow)..."
gh auth refresh -h github.com -s repo,read:org,workflow

echo "[gh] ensuring repo exists: ${owner}/${repo_name}"
if gh repo view "${owner}/${repo_name}" >/dev/null 2>&1; then
  echo "[gh] repo already exists"
  if ! git remote | rg -q '^origin$'; then
    git remote add origin "https://github.com/${owner}/${repo_name}.git"
  fi
  git push -u origin main
  exit 0
fi

echo "[gh] creating repo + pushing: ${owner}/${repo_name}"
gh repo create "${repo_name}" --public --source=. --remote=origin --push --description "${description}"
