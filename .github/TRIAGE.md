# Contributor triage

**Currently dry-run only.** No issues or PRs will be closed automatically.
Maintainers should inspect the CI logs to review proposed actions and decide
whether to adjust the workflow.

## Policy

- **Maintainers:** users with GitHub repository `write`, `maintain`, or `admin` access.
  Their submissions are exempt without an allowlist entry. They manage approvals by
  reviewing changes to [APPROVED_CONTRIBUTORS](APPROVED_CONTRIBUTORS); there is no
  comment-command approval workflow.
- **Approved contributors:** users on that list. Their submissions are also exempt,
  but approval grants no repository permissions or maintainer override authority.
- Both groups' comments, reviews, and issue-label activity count as contributor engagement.
  Names are case-insensitive; GitHub's automatic `CONTRIBUTOR` association is not approval.
- Bot/non-human authors are exempt; this policy does not close automated dependency PRs.

For everyone else:

- **Issues:** close 14 days after opening unless a contributor has commented or
  added any label. The exemption remains even if that label is removed.
- **PRs without engagement:** close 14 days after opening unless a contributor
  has commented (including inline) or submitted a GitHub review.
- **PRs with unanswered feedback:** remind 14 days after the latest contributor comment
  or submitted review. Close only if still unanswered seven days after the reminder was
  actually posted. New feedback starts a new cycle, not an immediate closure.
- An author comment, inline reply, submitted review, or request for review stops the response timer.
  Commits, edits, reactions, bots, and unrelated participants do not count as responses.
  Pending reviews are not engagement. Submitted reviews of any state count.
- A maintainer adding `keep-open` or reopening an item exempts it from inactivity closure.
  Removing the label does not undo the override. A non-maintainer reopening an item
  does not reset its timers; a later closure gets a fresh notice.
- Draft status, assignment, and locking do not exempt an item or reset its timers.

Roles and approvals are evaluated using current GitHub permissions and the current list,
not historical permissions. Removing access or approval may remove an exemption.
Permission lookups are cached within each scan. Lookup errors stop further writes to the
affected item and fail the run; they never mean someone is not a maintainer.

Replies stop inactivity closure; they do not mean review feedback is resolved. For
reconsideration or inclusion in the list, ask a maintainer in the issue/PR conversation.
Discussions are not locked and branches are not deleted.

The existing issue-policy workflow has separate scope checks. Passing its automated
check (or being exempt as an org member) does not count as human engagement here.

## Testing

Requires Node 24+ and, for the preview, an authenticated GitHub CLI (`gh`).

```sh
node --test .github/scripts/triage.test.mjs
node .github/scripts/triage.mjs --repo chalk-lab/Mooncake.jl
```

The tests use a fixed clock and synthetic API histories, including both sides of the
14-day and seven-day boundaries. Mocked writes exercise reminders, closures, reruns,
and activity arriving during a scan. No GitHub credentials or waits are needed for tests.

The preview performs GET requests only and prints each proposed action and its reason.
Without `--since YYYY-MM-DD`, it inspects the existing backlog for policy evaluation;
that does **not** mean the backlog will be included at launch. Add `--since` to preview
the exact rollout cutoff. API errors fail the command; completed writes are not rolled back.

## Enabling later

1. Review the allowlist and dry-run output. Publish the policy before enabling writes.
2. Set the repository Actions variable `TRIAGE_START_DATE` to the launch date (`YYYY-MM-DD`,
   UTC). Submissions before this date are exempt from the new automation.
3. Run **Contributor triage** manually with `apply` unchecked to preview that cutoff.
   For an end-to-end write smoke test, use a disposable repository, not Mooncake.
4. After approval, run manually with `apply` checked, or set `TRIAGE_APPLY=true` to enable
   scheduled writes. Unset it to return the daily schedule to dry-run. Manual runs always
   respect their own checkbox, even when scheduled writes are enabled.

Write mode requires GitHub Actions and an explicit cutoff. The workflow uses its
repository-scoped `GITHUB_TOKEN`, checks out the triage script and allowlist from the
default branch, serializes runs, and rechecks activity before writing and again
before closing. Reminder markers are accepted only from `github-actions[bot]`, not
copied human comments.

Runs are daily, but GitHub may delay them. GitHub disables public-repository
schedules after 60 days without repository activity. Actions occur on the next
successful run, never before their deadline. The tests run on relevant pull
requests with read-only permissions.
