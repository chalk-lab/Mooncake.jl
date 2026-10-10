# Contributor triage

**Currently dry-run only.** Review proposed actions in the CI logs. The workflow
does not post reminders or close issues or PRs unless writes are enabled.

## Policy

Contributors are the users in [contributors.txt](contributors.txt),
including maintainers. The current list alone determines contributor status;
names are case-insensitive, and inclusion grants no repository permissions.
Submissions by contributors and bots are exempt.

For everyone else:

- **Issues:** close 14 days after opening without a contributor comment or label.
  A label still counts after removal.
- **PRs:** close 14 days after opening without a contributor comment (including
  inline) or submitted review. All submitted review states count, including approvals.
- **Unanswered PR feedback:** remind 14 days after the latest contributor feedback;
  close if still unanswered seven full days after the reminder. New feedback
  starts a new cycle. Pending reviews do not count.
- **Author responses:** a comment, inline reply, submitted review, or review request
  stops the response timer until new feedback. Commits, edits, reactions, bots,
  and other participants do not count.
- **Overrides:** a listed contributor adding `keep-open` or reopening an item
  exempts it, even if the label is later removed. Other reopenings do not reset timers.

Deadlines use 24-hour days; daily runs may act later. Drafts, assignments, and locks
do not change the rules. For more time or reconsideration, ask a listed contributor
in the discussion. The existing issue-scope policy remains separate.

## Testing

Use Node 24+ and an authenticated GitHub CLI (`gh`) for the read-only preview:

```sh
node --test '.github/triage/test/*.test.mjs'
node .github/triage/cli.mjs --repo chalk-lab/Mooncake.jl
```

Tests use synthetic histories and need no credentials. The preview prints proposed
actions without writing; add `--since YYYY-MM-DD` to exclude older submissions.

## Layout

- `policy.mjs`: pure decision logic (`evaluate`) and its defaults (durations, override
  label, bot login). Portable to a GitHub App unchanged.
- `github.mjs`: scans open items via `gh api`, rechecks, then posts and closes.
- `cli.mjs`: the Actions entry point; parses flags and reads `contributors.txt`.
- `test/`: tests for each module above, with shared synthetic histories in `fixtures.mjs`.

## Enabling later

1. Review the contributor list and dry-run output; publish the policy before enforcement.
2. Set the Actions variable `TRIAGE_START_DATE` (`YYYY-MM-DD`, UTC). Older submissions
   are exempt. Preview with **Contributor triage** and `apply` unchecked.
3. After approval, check `apply` for a manual run or set `TRIAGE_APPLY=true` for
   scheduled writes. Unset the variable to restore scheduled previews; manual runs
   always follow their checkbox.

The workflow uses the default branch and `GITHUB_TOKEN`; no extra secret is needed.
Test actual reminders and closures in a disposable repository, not Mooncake.
