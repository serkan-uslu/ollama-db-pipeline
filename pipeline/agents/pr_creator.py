"""
pipeline/agents/pr_creator.py

F-05: GitHub PR Creator

Uses GitHub REST API via httpx (no PyGithub) to:
1. Get current models.json from target repo (+ sha for update)
2. Compare with new models.json — skip if no changes
3. Create/update branch with new file content (base64 encoded)
4. Open a Pull Request if one doesn't exist yet

Requires CROSS_REPO_TOKEN with `repo` scope.
"""

import base64
import json
import logging
from datetime import date
from pathlib import Path

import httpx

from pipeline.agents.exporter import DEFAULT_OUTPUT_PATH
from pipeline.core.settings import settings

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
TARGET_FILE_PATH = "public/models.json"  # path inside ollama-explorer repo


# ── HTTP Helpers ───────────────────────────────────────────────────────────────

def _headers() -> dict:
    if not settings.cross_repo_token:
        raise RuntimeError(
            "CROSS_REPO_TOKEN is not set. "
            "Add a GitHub PAT with repo scope to your .env file."
        )
    return {
        "Authorization": f"Bearer {settings.cross_repo_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    url = f"{GITHUB_API}{path}"
    resp = httpx.request(method, url, headers=_headers(), timeout=30, **kwargs)
    if resp.status_code >= 400:
        logger.error(f"GitHub API error {resp.status_code}: {resp.text[:300]}")
    return resp


# ── File Operations ────────────────────────────────────────────────────────────

def _get_file(repo: str, file_path: str, branch: str) -> tuple[str | None, str | None]:
    """
    Get file content and sha from a repo branch.
    Returns (content_str, sha) — both None if file doesn't exist.
    """
    resp = _api(
        "GET",
        f"/repos/{repo}/contents/{file_path}",
        params={"ref": branch},
    )
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def _put_file(
    repo: str,
    file_path: str,
    branch: str,
    content: str,
    sha: str | None,
    commit_message: str,
) -> bool:
    """Create or update a file in a repo branch. Returns True on success."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    payload: dict = {
        "message": commit_message,
        "content": encoded,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha  # required for updates

    resp = _api("PUT", f"/repos/{repo}/contents/{file_path}", json=payload)
    return resp.status_code in (200, 201)


def _ensure_branch(repo: str, branch: str, base_branch: str = "main") -> None:
    """Create branch if it doesn't exist. Silently skip if already exists."""
    # Get base branch SHA
    resp = _api("GET", f"/repos/{repo}/git/ref/heads/{base_branch}")
    if resp.status_code != 200:
        # Try 'master' fallback
        resp = _api("GET", f"/repos/{repo}/git/ref/heads/master")
        if resp.status_code != 200:
            raise RuntimeError(f"Cannot find base branch in {repo}")

    sha = resp.json()["object"]["sha"]

    # Try to create the branch
    create_resp = _api(
        "POST",
        f"/repos/{repo}/git/refs",
        json={"ref": f"refs/heads/{branch}", "sha": sha},
    )
    if create_resp.status_code == 422:
        logger.info(f"Branch '{branch}' already exists — will update in place.")
    elif create_resp.status_code == 201:
        logger.info(f"Branch '{branch}' created.")
    else:
        logger.warning(f"Unexpected response when creating branch: {create_resp.status_code}")


def _get_or_create_pr(repo: str, branch: str, base: str, title: str, body: str) -> str | None:
    """
    Check if a PR already exists for this branch.
    If yes — return its URL. If no — create it and return URL.
    """
    # Check existing PRs from this branch
    list_resp = _api(
        "GET",
        f"/repos/{repo}/pulls",
        params={"head": f"{repo.split('/')[0]}:{branch}", "state": "open"},
    )
    if list_resp.status_code == 200 and list_resp.json():
        existing = list_resp.json()[0]
        logger.info(f"PR already open: {existing['html_url']}")
        return existing["html_url"]

    # Create new PR
    create_resp = _api(
        "POST",
        f"/repos/{repo}/pulls",
        json={
            "title": title,
            "body": body,
            "head": branch,
            "base": base,
        },
    )
    if create_resp.status_code == 201:
        url = create_resp.json()["html_url"]
        logger.info(f"PR created: {url}")
        return url

    logger.error(f"Failed to create PR: {create_resp.status_code} {create_resp.text[:200]}")
    return None


# ── PR Body Builder ────────────────────────────────────────────────────────────

def _build_pr_body(new_data: list[dict], old_data: list[dict] | None, stats: dict) -> str:
    today = date.today().isoformat()

    old_ids = {m["model_identifier"] for m in (old_data or [])}
    new_ids = {m["model_identifier"] for m in new_data}

    new_count = len(new_ids - old_ids)
    updated_count = len(new_ids & old_ids)

    return f"""## 🦙 Ollama Models Update — {today}

**Changes:**
- Total models: {len(new_data)}
- New models: {new_count}
- Updated models: {updated_count}

**Pipeline stats:**
- Enriched: {stats.get('enriched', '?')}/{stats.get('total', '?')}
- Validated: {stats.get('validated', '?')}/{stats.get('total', '?')}
- Validation failed: {stats.get('validation_failed', 0)}
- Uncensored models: {stats.get('uncensored', 0)}

*Auto-generated by [ollama-pipeline](https://github.com/serkan-uslu/ollama-pipeline)*
"""


# ── Main Entry Point ───────────────────────────────────────────────────────────

def create_pull_request(
    models_json_path: str | Path = DEFAULT_OUTPUT_PATH,
    export_stats: dict | None = None,
) -> str | None:
    """
    1. Read local models.json
    2. Compare with current version in target repo
    3. Create/update branch + commit
    4. Open PR if changes detected

    Returns PR URL or None (if no changes or error).
    """
    models_json_path = Path(models_json_path)
    if not models_json_path.exists():
        raise FileNotFoundError(f"models.json not found at {models_json_path}")

    repo = settings.github_target_repo
    branch = settings.github_target_branch
    today = date.today().isoformat()

    # Read new models.json
    new_content = models_json_path.read_text(encoding="utf-8")
    new_data = json.loads(new_content)
    logger.info(f"Loaded {len(new_data)} models from {models_json_path}")

    # Get current file from repo (main branch first)
    old_content, old_sha = _get_file(repo, TARGET_FILE_PATH, "main")
    if old_sha is None:
        old_content, old_sha = _get_file(repo, TARGET_FILE_PATH, "master")

    old_data = json.loads(old_content) if old_content else None

    # Compare — skip if identical
    if old_content and old_content.strip() == new_content.strip():
        logger.info("No changes detected — skipping PR.")
        return None

    # Ensure target branch exists
    _ensure_branch(repo, branch)

    # Get sha of file on the target branch (for update)
    _, branch_sha = _get_file(repo, TARGET_FILE_PATH, branch)

    # Commit updated file
    commit_msg = f"chore: update models data — {today} ({len(new_data)} models)"
    success = _put_file(
        repo=repo,
        file_path=TARGET_FILE_PATH,
        branch=branch,
        content=new_content,
        sha=branch_sha,
        commit_message=commit_msg,
    )
    if not success:
        logger.error("Failed to push models.json to branch.")
        return None

    logger.info(f"Committed {len(new_data)} models to branch '{branch}'")

    # Build PR
    pr_title = f"chore: update ollama models — {today}"
    pr_body = _build_pr_body(new_data, old_data, export_stats or {})

    return _get_or_create_pr(
        repo=repo,
        branch=branch,
        base="main",
        title=pr_title,
        body=pr_body,
    )
