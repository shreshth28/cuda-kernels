#!/usr/bin/env bash
# setup.sh — one-time project bootstrap

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()     { echo -e "${RED}[error]${NC} $*"; exit 1; }
section() { echo -e "\n${YELLOW}━━━ $* ━━━${NC}"; }

# ── 1. Python dependencies ─────────────────────────────────────────────────
section "Python dependencies"

if ! command -v python3 &>/dev/null; then
    die "python3 not found. Install Python 3.9+ first."
fi

python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt
info "Python packages installed."

# ── 2. GitHub repo check ────────────────────────────────────────────────────
section "GitHub repository"

REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [[ -z "$REMOTE_URL" ]]; then
    warn "No git remote 'origin' found."
    warn "Create a GitHub repo and run:  git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git"
else
    info "Remote: $REMOTE_URL"
fi

# ── 3. GitHub secrets instructions ─────────────────────────────────────────
section "GitHub Secrets required"

cat <<'EOF'

Add these secrets at:
  https://github.com/YOUR_USER/YOUR_REPO/settings/secrets/actions

  GH_PAT
    A GitHub Personal Access Token with "repo" scope.
    Create at: https://github.com/settings/tokens?type=beta
    Needs: Contents (read & write), Commit statuses (write)

  GOOGLE_SERVICE_ACCOUNT_JSON   (optional — enables Drive notebook annotation)
    The full JSON of a Google Cloud service account key.
    Steps:
      1. Create project at https://console.cloud.google.com/
      2. IAM & Admin → Service Accounts → Create Service Account
      3. Keys → Add Key → JSON  (download the file)
      4. APIs & Services → Enable "Google Drive API"
      5. Share your Colab notebook with the service account email (Viewer is enough)
      6. Paste the entire JSON content as the secret value

  NOTEBOOK_ID                   (optional — paired with above)
    The Google Drive file ID of auto_runner.ipynb.
    Find it in the notebook URL: drive.google.com/file/d/NOTEBOOK_ID/...

EOF

# ── 4. Upload notebook to Google Drive ─────────────────────────────────────
section "Colab notebook"

cat <<'EOF'

Upload auto_runner.ipynb to Google Drive:
  1. Go to https://drive.google.com
  2. Upload auto_runner.ipynb  (or open https://colab.research.google.com and import it)
  3. Open the notebook in Colab
  4. Runtime → Change runtime type → GPU (T4 is free)
  5. Open the 🔑 Secrets sidebar and add:
       GH_PAT       = ghp_...
       GITHUB_REPO  = your-username/your-repo
  6. Runtime → Run all

The notebook will poll for new jobs every 10 seconds.
Leave the Colab tab open — it runs until the ~12 h free-tier limit.

EOF

# ── 5. git init & first push ────────────────────────────────────────────────
section "Git setup"

if [[ ! -d ".git" ]]; then
    git init
    git add .
    git commit -m "Initial commit: CUDA auto-execution pipeline"
    info "Git repo initialized."
    warn "Now push to GitHub:  git remote add origin URL && git push -u origin main"
else
    info "Git repo already initialized."
    info "Push to trigger the first workflow:  git push"
fi

# ── 6. Validate jobs/ and results/ placeholder files ───────────────────────
section "Placeholder files"

for dir in jobs results; do
    if [[ ! -f "$dir/.gitkeep" ]]; then
        touch "$dir/.gitkeep"
        git add "$dir/.gitkeep" 2>/dev/null || true
        info "Created $dir/.gitkeep"
    fi
done

echo ""
info "Setup complete! See README.md for the full workflow."
