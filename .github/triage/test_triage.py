#!/usr/bin/env python3
"""Offline integration tests for triage.sh; requires bash, GNU date and jq."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone

DAY = 86400
NOW = int(datetime(2026, 10, 22, tzinfo=timezone.utc).timestamp())
SCRIPT = Path(os.environ.get("TRIAGE_SCRIPT", ".github/triage/triage.sh")).resolve()
MARKER = "<!-- triage:reminder -->"


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def item(pr=True, created=NOW - 40 * DAY, number=1, author="author", labels=(), kind="User"):
    result = dict(number=number, created_at=stamp(created),
                  user=dict(login=author, type=kind), labels=[dict(name=x) for x in labels])
    if pr:
        result["pull_request"] = {}
    return result


def event(kind="commented", who="maintainer", at=NOW - 14 * DAY, **fields):
    result = dict(event=kind, html_url="https://example.test/feedback", **fields)
    result["user" if kind in ("commented", "reviewed") else "actor"] = dict(login=who)
    result["submitted_at" if kind == "reviewed" else "created_at"] = stamp(at)
    return result


def reminder(at, who="github-actions[bot]", body=MARKER):
    return event(who=who, at=at, body=body)


def mock_gh():
    """Serve raw API pages, applying the actual --jq filter; record every call."""
    args = sys.argv[1:]
    with open("calls.jsonl", "a") as out:
        out.write(json.dumps(args) + "\n")
    data = json.loads(Path("fixture.json").read_text())
    if args[0] != "api":
        assert args[:2] in (["pr", "comment"], ["pr", "close"], ["issue", "close"])
        if data.get("fail") == "write":
            sys.exit(1)
        return
    endpoint = args[1]
    if "/timeline?" in endpoint:
        kind = "timeline"
    elif "/comments?" in endpoint:
        kind = "comments"
    elif endpoint == "repos/test/repo/issues?state=open&per_page=100":
        kind = "items"
    else:
        raise AssertionError(endpoint)
    number = re.search(r"/(\d+)/", endpoint)
    pages = data[kind] if kind == "items" else data.get(kind, {}).get(number[1], [[]])
    assert "--paginate" in args
    if kind != "items":
        assert "--slurp" in args
    for index, page in enumerate(pages):
        if data.get("fail") == kind and index == data.get("fail_page", 0):
            print("mock API error", file=sys.stderr)
            sys.exit(1)
        if kind == "items":
            query = args[args.index("--jq") + 1]
            subprocess.run(["jq", "-r", query], input=json.dumps(page), text=True, check=True)
    if kind != "items":
        print(json.dumps(pages))


class TriageTests(unittest.TestCase):
    def run_bot(self, items=None, events=(), comments=(), now=NOW, apply=True,
                start="1970-01-01", approved="  Maintainer # trusted\n", **extra):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".github").mkdir()
            (root / ".github/APPROVED_CONTRIBUTORS").write_text(approved)
            (root / "bin").mkdir()
            source = Path(__file__).read_text().split("\n", 1)[1]
            for name in ("gh", "date"):
                path = root / "bin" / name
                path.write_text(f"#!{sys.executable}\n" + source)
                path.chmod(0o755)
            fixture = dict(items=items if items is not None else [[item()]],
                           timeline={"1": [list(events)]},
                           comments=comments if isinstance(comments, dict) else {"1": [list(comments)]})
            fixture.update(extra)
            (root / "fixture.json").write_text(json.dumps(fixture))
            env = dict(PATH=f"{root / 'bin'}:/usr/bin:/bin", LANG="C.UTF-8",
                       TZ="Pacific/Honolulu", MOCK_NOW=str(now), GH_REPO="test/repo",
                       GH_TOKEN="unused-offline", TRIAGE_START_DATE=start,
                       TRIAGE_APPLY="true" if apply else "false")
            result = subprocess.run(["bash", str(SCRIPT)], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=20)
            calls = [json.loads(x) for x in (root / "calls.jsonl").read_text().splitlines()] \
                if (root / "calls.jsonl").exists() else []
            return result, calls

    def check(self, expected, *, failure=False, **kwargs):
        result, calls = self.run_bot(**kwargs)
        writes = [x for x in calls if x[0] != "api"]
        if failure:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(writes, [])
            return result, calls
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"#1: {expected} (", result.stdout)
        expected_write = kwargs.get("apply", True) and expected != "keep"
        self.assertEqual(len(writes), int(expected_write), writes)
        if expected_write:
            pr = "pull_request" in kwargs.get("items", [[item()]])[0][0]
            self.assertEqual(writes[0][:3], ["pr" if pr else "issue",
                             "comment" if expected == "remind" else "close", "1"])
            self.assertIn("triage policy", writes[0][-1])
            if expected == "remind":
                self.assertIn(MARKER, writes[0][-1])
                self.assertIn("https://example.test/feedback", writes[0][-1])
        return result, calls

    def test_no_response_boundary_and_preview(self):
        for pr in (False, True):
            for offset, expected in ((-1, "keep"), (0, "close"), (1, "close")):
                for apply in (False, True):
                    with self.subTest(pr=pr, offset=offset, apply=apply):
                        self.check(expected, items=[[item(pr=pr, created=NOW - 14 * DAY - offset)]],
                                   apply=apply)

    def test_feedback_boundary_and_preview(self):
        for offset, expected in ((-1, "keep"), (0, "remind"), (1, "remind")):
            for apply in (False, True):
                with self.subTest(offset=offset, apply=apply):
                    self.check(expected, events=[event(at=NOW - 14 * DAY - offset)], apply=apply)

    def test_reminder_lifecycle(self):
        feedback = event(at=NOW - 30 * DAY)
        _, calls = self.check("remind", events=[feedback])
        body = [x for x in calls if x[0] != "api"][0][-1]
        events = [feedback, reminder(NOW, body=body)]
        for offset, expected in ((0, "keep"), (7 * DAY - 1, "keep"),
                                 (7 * DAY, "close"), (7 * DAY + 1, "close")):
            for apply in (False, True):
                with self.subTest(offset=offset, apply=apply):
                    self.check(expected, events=events, now=NOW + offset, apply=apply)

    def test_author_activity(self):
        feedback = event(at=NOW - 30 * DAY)
        for kind in ("commented", "reviewed", "review_requested", "reopened", "committed"):
            for offset in (-1, 0, 1):
                with self.subTest(kind=kind, offset=offset):
                    reply = event(kind, "AuThOr", NOW - 30 * DAY + offset)
                    expected = "keep" if kind != "committed" and offset >= 0 else "remind"
                    self.check(expected, events=[feedback, reply])
        self.check("remind", events=[feedback, event(who="bystander", at=NOW)])
        # Inline replies come from the separate pull-request comments endpoint.
        reply = event(who="author", at=NOW)
        del reply["event"]
        self.check("keep", events=[feedback, reminder(NOW - 7 * DAY)], comments=[reply])
        self.check("close", events=[feedback, reminder(NOW - 7 * DAY),
                                    event("committed", "author", NOW)])

    def test_issue_responses_and_pr_labels(self):
        for kind in ("commented", "reviewed", "labeled"):
            with self.subTest(kind=kind):
                self.check("keep", items=[[item(pr=False)]], events=[event(kind)])
        self.check("close", events=[event("labeled")])
        self.check("close", items=[[item(pr=False)]], events=[event(who="bystander")])
        inline = event()
        del inline["event"]
        self.check("remind", comments=[inline])

    def test_new_feedback_and_approvals(self):
        old = event(at=NOW - 40 * DAY)
        for kind in ("commented", "reviewed"):
            with self.subTest(kind=kind):
                new = event(kind, at=NOW - 10 * DAY, state="changes_requested")
                events = [old, reminder(NOW - 26 * DAY), event(who="author", at=NOW - 20 * DAY), new]
                self.check("keep", events=events)
                self.check("remind", events=events, now=NOW + 4 * DAY)
        approval = event("reviewed", state="approved", at=NOW - 30 * DAY)
        self.check("keep", events=[approval])
        self.check("remind", events=[approval, event()])
        self.check("keep", events=[old, approval])

    def test_approval_with_simultaneous_inline_comment(self):
        approval = event("reviewed", state="approved", at=NOW - 30 * DAY)
        comment = event(at=NOW - 30 * DAY)
        del comment["event"]
        for events in ([approval], [approval, reminder(NOW - 7 * DAY)]):
            with self.subTest(events=events):
                self.check("keep", events=events, comments=[comment])

    def test_reminder_matching(self):
        feedback = event(at=NOW - 30 * DAY)
        for invalid in (reminder(NOW - 31 * DAY), reminder(NOW - 10 * DAY, who="author"),
                        reminder(NOW - 10 * DAY, who="other[bot]"),
                        reminder(NOW - 10 * DAY, body="ordinary bot comment")):
            with self.subTest(invalid=invalid):
                # A real author comment is a reply regardless of marker.
                self.check("keep" if invalid["user"]["login"] == "author" else "remind",
                           events=[feedback, invalid])
        self.check("close", events=[feedback, reminder(NOW - DAY), reminder(NOW - 8 * DAY)])

    def test_exemptions_and_utc_cutoff(self):
        for pr in (False, True):
            for changes in (dict(author="MAINTAINER"), dict(kind="Bot"), dict(labels=["keep-open"])):
                with self.subTest(pr=pr, changes=changes):
                    self.check("keep", items=[[item(pr=pr, **changes)]])
            self.check("keep", items=[[item(pr=pr)]], events=[event("reopened")])
        cutoff = NOW - 20 * DAY
        for offset, expected in ((-1, "keep"), (0, "close"), (1, "close")):
            with self.subTest(offset=offset):
                self.check(expected, items=[[item(created=cutoff + offset)]], start="2026-10-02")
        self.check("close", start="", apply=False)

    def test_configuration_errors(self):
        for start in ("", "2026-02-30", "2026-1-01", "yesterday"):
            with self.subTest(start=start):
                _, calls = self.check(None, start=start, failure=True)
                self.assertEqual(calls, [])
        for approved in ("", "# empty\n", "not a username\n"):
            with self.subTest(approved=approved):
                self.check(None, approved=approved, failure=True)

    def test_pagination_and_failures(self):
        # Put the only response / inline reply on the second page.
        self.check("keep", timeline={"1": [[], [event()]]}, items=[[item(pr=False)]])
        reply = event(who="author", at=NOW)
        del reply["event"]
        self.check("keep", events=[event()], comments={"1": [[], [reply]]})
        result, calls = self.run_bot(items=[[item(pr=False)], [item(pr=False, number=2)]])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#2: close", result.stdout)
        self.assertEqual([x[2] for x in calls if x[0] == "issue"], ["1", "2"])
        self.assertEqual(calls[0][0], "api")
        for kind in ("items", "timeline", "comments"):
            for page in (0, 1):
                with self.subTest(kind=kind, page=page):
                    overrides = {kind: [[item()], []]} if kind == "items" else {kind: {"1": [[], []]}}
                    self.check(None, failure=True, fail=kind, fail_page=page, **overrides)
        for invalid in ({"unexpected": True}, event(at=NOW) | {"created_at": "invalid"}):
            with self.subTest(invalid=invalid):
                self.check(None, failure=True, events=[invalid])
        for first, events in ((item(pr=False), []), (item(), []), (item(), [event()])):
            with self.subTest(first=first, events=events):
                result, calls = self.run_bot(items=[[first, item(number=2)]], events=events,
                                             fail="write")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len([x for x in calls if x[0] != "api"]), 1)
                self.assertNotIn("#2:", result.stdout)


if __name__ == "__main__":
    if Path(sys.argv[0]).name == "gh":
        mock_gh()
    elif Path(sys.argv[0]).name == "date":
        if sys.argv[1:] == ["+%s"]:
            print(os.environ["MOCK_NOW"])
        else:
            os.execv("/usr/bin/date", ["date", *sys.argv[1:]])
    else:
        unittest.main()
