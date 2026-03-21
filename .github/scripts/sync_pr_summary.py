#!/usr/bin/env python3

import json
import os
import subprocess
import sys


START_MARKER = "<!-- managed-pr-summary:start -->"
END_MARKER = "<!-- managed-pr-summary:end -->"

COMMENT_MARKERS = {
    "<!-- docs-preview-url-Mooncake.jl -->": (
        "Documentation Preview",
        "_Pending. No docs preview comment found yet._",
    ),
    "<!-- perf-results -->": (
        "Performance",
        "_Pending. No performance comment found yet._",
    ),
}

SECTION_ENV_VARS = {
    "Documentation Preview": "SUMMARY_DOCS_CONTENT",
    "Performance": "SUMMARY_PERF_CONTENT",
}


def gh_api(*args: str) -> str:
    proc = subprocess.run(
        ["gh", "api", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def find_managed_block(body: str) -> tuple[int, int] | None:
    start = body.find(START_MARKER)
    end = body.find(END_MARKER)
    if start == -1 and end == -1:
        return None
    if start != -1 and (end == -1 or end < start):
        return start, len(body)
    if start == -1 and end != -1:
        return 0, end + len(END_MARKER)
    return start, end + len(END_MARKER)


def extract_sections(comments: list[dict]) -> list[str]:
    latest_by_marker: dict[str, str] = {}
    for comment in comments:
        body = comment.get("body") or ""
        user = (comment.get("user") or {}).get("login")
        app = (comment.get("performed_via_github_app") or {}).get("slug")
        if user != "github-actions[bot]" and app != "github-actions":
            continue
        for marker, (title, _) in COMMENT_MARKERS.items():
            if body.startswith(marker):
                content = body[len(marker) :].strip()
                if content:
                    latest_by_marker[marker] = f"### {title}\n\n{content}"

    sections = []
    for marker, (title, fallback) in COMMENT_MARKERS.items():
        override = os.environ.get(SECTION_ENV_VARS[title])
        if override is not None:
            sections.append(f"### {title}\n\n{override.strip()}")
            continue
        sections.append(
            latest_by_marker.get(marker, f"### {title}\n\n{fallback}")
        )
    return sections


def build_updated_body(base_body: str, sections: list[str]) -> str:
    managed_block = "\n\n".join(
        [
            START_MARKER,
            "## Automated PR Summary",
            "",
            *sections,
            END_MARKER,
        ]
    )

    block_span = find_managed_block(base_body)
    if block_span is not None:
        start, end = block_span
        return f"{base_body[:start]}{managed_block}{base_body[end:]}"

    if not base_body:
        return managed_block
    separator = ""
    if not base_body.endswith(("\n", " ", "\t")):
        separator = "\n\n"
    elif not base_body.endswith("\n\n"):
        separator = "\n"
    return f"{base_body}{separator}{managed_block}"


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")
    if not repo or not pr_number:
        print("GITHUB_REPOSITORY and PR_NUMBER must be set", file=sys.stderr)
        return 1

    pr = json.loads(gh_api(f"repos/{repo}/pulls/{pr_number}"))
    comments = json.loads(gh_api("--paginate", f"repos/{repo}/issues/{pr_number}/comments"))

    current_body = pr.get("body") or ""
    sections = extract_sections(comments)
    updated_body = build_updated_body(current_body, sections)

    if updated_body == current_body:
        print("PR summary already up to date.")
        return 0

    gh_api(
        "--method",
        "PATCH",
        f"repos/{repo}/pulls/{pr_number}",
        "-f",
        f"body={updated_body}",
    )
    print("Updated PR summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
