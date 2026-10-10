# How we follow up on issues and pull requests

> **Not active yet.** The triage bot currently only previews what it would do; it
> does not post comments or close anything.

Mooncake is maintained part time by a small team. Our aim is a solid package that
serves almost all users and use cases, with a careful trade-off between complexity
and features. We want the list of open issues and pull requests to be a fair picture
of what is actually being worked on, so a bot closes items that have stalled. This
page explains when, and how to make sure your work stays open.

Closing an issue or PR doesn't mean we've rejected it. If you'd like it reopened,
leave a comment and a maintainer can reopen it.

## Approved contributors

People listed in [APPROVED_CONTRIBUTORS](../APPROVED_CONTRIBUTORS) are trusted
to see their work through. Their issues and pull requests are never closed by the
bot, and their comments, labels and reviews count as a response on anyone else's.
The list includes the maintainers.

If you would like to contribute regularly and can commit time to follow up on
reviews, get in touch with Xianda Sun ([@sunxd3](https://github.com/sunxd3)) by email
or on the [Julia Slack](https://julialang.org/slack/), and we will consider adding you.

The rules below apply to everyone else.

## If no one on the list has responded

If an issue or pull request has had no response from an approved contributor within
14 days of being opened, it is closed. This reflects our limited capacity, not the
value of the submission.

## If your pull request has been reviewed

Once an approved contributor leaves feedback, the next step is usually yours:

- If 14 days pass without a reply, the bot posts a friendly reminder.
- If there is still no reply 7 days after that, the pull request is closed to keep
  the review queue clean.

Any of these counts as a reply: a comment, a reply to a review comment, a review, or
re-requesting review. Pushing commits on its own does not, because we cannot tell
from commits whether the feedback has been addressed. A short "still working on
this" is plenty, and new feedback starts the clock again.

## Need more time?

On a reviewed pull request, say so in a comment; that counts as a reply. Otherwise,
ask in a comment and an approved contributor can add the `keep-open` label (or
reopen it if it was closed), which tells the bot to leave it alone for good.
