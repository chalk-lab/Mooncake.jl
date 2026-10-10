# Maintaining the triage bot

The public policy is in [README.md](README.md); the bot's comments link to it, so
keep it in sync with any behaviour change here.

## Configuration

- [approved-contributors.txt](approved-contributors.txt): whose own submissions are
  exempt, and whose comments, labels and reviews count as a response. Include all
  maintainers. Listing grants no repository permissions. One GitHub username per
  line, case-insensitive; `#` starts a comment. Add people by PR.
- Durations, the `keep-open` label name and the bot login are defaults in
  `policy.mjs` (`DEFAULTS`).

## Layout

- `policy.mjs`: pure decision logic (`evaluate`): `keep`, `remind` or `close`.
  Portable to a GitHub App unchanged.
- `github.mjs`: scans open items via `gh api` and posts or closes. Each bot comment
  carries a hidden `mooncake-triage:v1:<action>:<key>` marker, which is how the bot
  recognises its own earlier notices.
- `cli.mjs`: the Actions entry point; parses flags and reads `approved-contributors.txt`.
- `test/`: tests for each module above, with shared synthetic histories in
  `fixtures.mjs`.

## Testing

Use Node 24+ and an authenticated GitHub CLI (`gh`) for the read-only preview:

```sh
node --test '.github/triage/test/*.test.mjs'
node .github/triage/cli.mjs --repo chalk-lab/Mooncake.jl
```

Tests use synthetic histories and need no credentials. The preview prints proposed
actions without writing; add `--since YYYY-MM-DD` to exclude older submissions.
Test real comments and closures in a disposable repository, not Mooncake.

## Enabling

1. Review the approved-contributor list and the dry-run output. Create the `keep-open` label,
   remove the "Not active yet" note from README.md, and announce the policy.
2. Set the Actions variable `TRIAGE_START_DATE` (`YYYY-MM-DD`, UTC). Older submissions
   are exempt; use `1970-01-01` to include everything. Preview with
   **Contributor triage** and `apply` unchecked.
3. Check `apply` for a manual run, then set `TRIAGE_APPLY=true` for scheduled writes.
   Unset the variable to return to scheduled previews; manual runs always follow
   their checkbox.

The workflow uses the default branch and `GITHUB_TOKEN`; no extra secret is needed.
