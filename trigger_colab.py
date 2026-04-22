#!/usr/bin/env python3
"""
trigger_colab.py — Submit a CUDA kernel job and wait for Colab GPU results.

Job flow:
  1. Write jobs/current.json to the GitHub repo (the Colab notebook polls this).
  2. Optionally update the Colab notebook in Google Drive via service account.
  3. Poll results/{sha}.txt until the notebook pushes output back.
  4. Write results to --output-file for the GitHub Action to read.
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# GitHub REST helpers
# ---------------------------------------------------------------------------

def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_get_file(token: str, repo: str, path: str):
    """Return (decoded_content, sha) or (None, None) if 404."""
    import requests
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token),
        timeout=30,
    )
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode()
    return content, data["sha"]


def github_put_file(token: str, repo: str, path: str, content: str, message: str, sha: str | None = None):
    """Create or update a file in the repo."""
    import requests
    payload: dict = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Google Drive helper (optional — annotates the notebook with current job)
# ---------------------------------------------------------------------------

def update_drive_notebook(notebook_id: str, cu_file: str, sha: str):
    """
    Update a property in the Google Drive notebook file so the Colab UI
    shows which job is active. Requires GOOGLE_SERVICE_ACCOUNT_JSON env var.

    This is optional — the notebook uses GitHub polling as the primary signal.
    """
    try:
        import json as _json
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not sa_json or not notebook_id:
            return  # silently skip if not configured

        sa_info = _json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/drive.metadata.readonly"],
        )
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)

        # Update the notebook description to reflect the active job
        drive.files().update(
            fileId=notebook_id,
            body={"description": f"Active job: {cu_file} | sha={sha[:8]}"},
        ).execute()
        print(f"[drive] Notebook {notebook_id} description updated.")
    except Exception as exc:
        # Drive update is best-effort; don't block the pipeline
        print(f"[drive] Skipped Drive update: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def submit_job(token: str, repo: str, sha: str, cu_file: str):
    job = {
        "sha": sha,
        "cu_file": cu_file,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    _, existing_sha = github_get_file(token, repo, "jobs/current.json")
    github_put_file(
        token, repo,
        path="jobs/current.json",
        content=json.dumps(job, indent=2) + "\n",
        message=f"[ci skip] job: {cu_file} ({sha[:8]})",
        sha=existing_sha,
    )
    print(f"[job] Submitted: {cu_file}  sha={sha[:8]}")


def wait_for_results(token: str, repo: str, sha: str, timeout: int, poll_interval: int = 15) -> str | None:
    results_path = f"results/{sha}.txt"
    deadline = time.monotonic() + timeout

    print(f"[poll] Waiting for {results_path} (timeout={timeout}s, interval={poll_interval}s)")

    while time.monotonic() < deadline:
        content, _ = github_get_file(token, repo, results_path)
        if content is not None:
            print("[poll] Results received.")
            return content
        remaining = int(deadline - time.monotonic())
        print(f"[poll] Not ready — {remaining}s left...")
        time.sleep(poll_interval)

    return None


def main():
    parser = argparse.ArgumentParser(description="Trigger Colab CUDA execution")
    parser.add_argument("--cu-file",     required=True,  help="Repo-relative path, e.g. kernels/matmul_naive.cu")
    parser.add_argument("--sha",         required=True,  help="Git commit SHA")
    parser.add_argument("--timeout",     type=int, default=600, help="Max seconds to wait (default: 600)")
    parser.add_argument("--output-file", default="results.txt", help="Where to write results for the Action")
    args = parser.parse_args()

    token = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GH_PAT or GITHUB_TOKEN env var not set.", file=sys.stderr)
        sys.exit(1)

    repo = os.environ.get("GITHUB_REPO")
    if not repo:
        print("ERROR: GITHUB_REPO env var not set (e.g. 'user/cuda-kernels').", file=sys.stderr)
        sys.exit(1)

    notebook_id = os.environ.get("NOTEBOOK_ID", "")

    # 1. Optionally annotate the Drive notebook
    update_drive_notebook(notebook_id, args.cu_file, args.sha)

    # 2. Write the job file so the Colab notebook picks it up
    submit_job(token, repo, args.sha, args.cu_file)

    # 3. Poll for results
    results = wait_for_results(token, repo, args.sha, timeout=args.timeout)

    output_path = Path(args.output_file)
    if results is None:
        msg = (
            f"ERROR: Timed out after {args.timeout}s waiting for Colab results.\n"
            f"Make sure the auto_runner notebook is running on a Colab GPU session.\n"
            f"Expected: results/{args.sha}.txt"
        )
        output_path.write_text(msg)
        print(msg, file=sys.stderr)
        sys.exit(1)

    output_path.write_text(results)
    print(f"\n[done] Results written to {args.output_file}")
    print("\n--- OUTPUT ---")
    print(results)


if __name__ == "__main__":
    main()
