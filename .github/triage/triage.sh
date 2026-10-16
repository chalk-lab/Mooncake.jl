#!/usr/bin/env bash
# Contributor triage (see README.md). Run by .github/workflows/triage.yml; needs gh and jq,
# and GH_TOKEN, GH_REPO, TRIAGE_START_DATE and TRIAGE_APPLY in the environment.
set -euo pipefail
if [[ $TRIAGE_APPLY == true && -z $TRIAGE_START_DATE ]]; then
  echo "::error::Applying requires the TRIAGE_START_DATE variable"; exit 1
fi
if [[ -n $TRIAGE_START_DATE && ! $TRIAGE_START_DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "::error::TRIAGE_START_DATE must be YYYY-MM-DD"; exit 1
fi
since=$(date -ud "${TRIAGE_START_DATE:-1970-01-01}" +%s)
names=$(sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//' .github/APPROVED_CONTRIBUTORS | grep -v '^$' || true)
if [[ -z $names ]] || grep -qvE '^[A-Za-z0-9][A-Za-z0-9-]*$' <<<"$names"; then
  echo "::error::.github/APPROVED_CONTRIBUTORS must list GitHub usernames, one per line"; exit 1
fi
approved=$(jq -Rn '[inputs | ascii_downcase]' <<<"$names")

# Reads one item's timeline and prints "ACTION FEEDBACK_URL REASON".
decide='
  def approved: .who | IN($approved[]);
  def keep(reason): "keep - \(reason)";
  (24 * 60 * 60) as $day
  | ($item.user.login // "" | ascii_downcase) as $author
  | ($item.created_at | fromdate) as $created
  | (flatten | map(if .event then . else error("unexpected timeline entry: \(tojson[:200])") end | {
      event, body, state, label: .label.name, url: .html_url,
      who: ((.user // .actor).login // "" | ascii_downcase),
      at: ((.submitted_at // .created_at // empty) | fromdate)
    })) as $events
  | if $created < $since then keep("opened before the start date")
    elif $item.user.type != "User" then keep("bot author")
    elif ($author | IN($approved[])) then keep("approved author")
    elif any($item.labels[]; . == "keep-open") then keep("labelled keep-open")
    elif any($events[]; approved and .event == "reopened") then keep("reopened by an approved contributor")
    else
      # On issues a label from an approved contributor counts as a response too.
      ([$events[] | select(approved and (.event | IN("commented", "reviewed", if $item.pr then empty else "labeled" end)))]
        | max_by(.at)) as $feedback
      | if $feedback == null then
          if $now < $created + 14 * $day then keep("opened less than 14 days ago")
          else "close - no response from an approved contributor in 14 days" end
        elif $item.pr | not then keep("an approved contributor has responded")
        elif $feedback.event == "reviewed" and $feedback.state == "approved" then keep("approved; waiting on maintainers")
        elif any($events[]; .who == $author and .at >= $feedback.at
            and (.event | IN("commented", "reviewed", "review_requested", "reopened")))
          then keep("author replied; waiting on approved contributors")
        elif $now < $feedback.at + 14 * $day then keep("feedback is less than 14 days old")
        else
          ([$events[] | select(.event == "commented" and .who == "github-actions[bot]" and .at >= $feedback.at
              and (.body // "" | contains("<!-- triage:reminder -->")))] | min_by(.at)) as $reminder
          | if $reminder == null then "remind \($feedback.url) feedback unanswered for 14 days"
            elif $now < $reminder.at + 7 * $day then keep("reminded less than 7 days ago")
            else "close \($feedback.url) no reply 7 days after the reminder" end
        end
    end'

note=$'\n\n'"<sub>Posted automatically under the [triage policy](https://github.com/$GH_REPO/blob/HEAD/.github/triage/README.md).</sub>"
# Fetch the full list first, so closing items cannot shift later pages.
items=$(gh api "repos/$GH_REPO/issues?state=open&per_page=100" --paginate \
  --jq '.[] | {number, created_at, user: {login: .user.login, type: .user.type}, labels: [.labels[].name], pr: has("pull_request")} | @json')
while read -r item; do
  [[ -n $item ]] || continue
  number=$(jq -r .number <<<"$item")
  events=$(gh api "repos/$GH_REPO/issues/$number/timeline?per_page=100" --paginate --slurp)
  if [[ $(jq -r .pr <<<"$item") == true ]]; then
    # Replies in review threads are not in the timeline; add them as comments.
    replies=$(gh api "repos/$GH_REPO/pulls/$number/comments?per_page=100" --paginate --slurp)
    events=$(jq -s '[.[0], (.[1] | flatten | map(.event = "commented"))]' <<<"$events $replies")
  fi
  read -r action url reason < <(jq -r --argjson item "$item" --argjson approved "$approved" \
    --argjson since "$since" --argjson now "$(date +%s)" "$decide" <<<"$events")
  echo "#$number: $action ($reason)"
  if [[ $action == keep || $TRIAGE_APPLY != true ]]; then continue; fi

  if [[ $action == remind ]]; then
    gh pr comment "$number" --body "Hi @$(jq -r .user.login <<<"$item"), a friendly reminder that there is [review feedback]($url) on this PR waiting for your reply. When you get a chance, please leave a comment (even a short \"still working on it\" is enough) or re-request review once it is ready; we cannot tell from commits alone whether feedback has been addressed. If there is no reply within 7 days, this PR will be closed to keep the review queue manageable. Nothing is lost if that happens, and you can ask for it to be reopened later.$note"$'\n\n<!-- triage:reminder -->' </dev/null
    continue
  fi
  if [[ $url == - ]]; then
    why="we have not been able to respond within 14 days. That reflects our limited capacity, not a judgment on the report or fix"
  else
    why="the [review feedback]($url) has not had a reply for a while. This is not a judgment on the work"
  fi
  comment="Closing this for now, as $why. Nothing is lost: the discussion stays here. If you think it should be reopened, say why in a comment in your own words and a maintainer will take a look.$note"
  if [[ $(jq -r .pr <<<"$item") == true ]]; then
    gh pr close "$number" --comment "$comment" </dev/null
  else
    gh issue close "$number" --reason "not planned" --comment "$comment" </dev/null
  fi
done <<<"$items"
