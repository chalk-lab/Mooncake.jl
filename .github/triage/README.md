# How we follow up on issues and pull requests

Mooncake is maintained part time by a small team. Our aim is a solid package that
serves almost all users and use cases, with a careful trade-off between complexity
and features. We want the list of open issues and pull requests to be a fair picture
of what is actually being worked on, so a bot closes items that have stalled. This
page explains when, and how to make sure your work stays open.

Closing an issue or PR doesn't mean we've rejected it. If you think it should be
reopened, leave a comment saying why, in your own words rather than purely
AI-generated. A maintainer will look at your comment and the issue or PR itself, and
decide whether to reopen it.

We start with a short list of people we already work with, and expect it to grow as
more people contribute regularly. People listed in
[APPROVED_CONTRIBUTORS](../APPROVED_CONTRIBUTORS) are trusted to see their work
through. Their issues and pull requests are never closed by the bot, and their
comments and reviews (on issues, labels too) count as a response on anyone else's.
The list includes the maintainers. If you would like to contribute regularly and can
commit time to follow up on reviews, we would be glad to hear from you: get in touch
with Xianda Sun ([@sunxd3](https://github.com/sunxd3)) by email or on the
[Julia Slack](https://julialang.org/slack/), and we will consider adding you.

For everyone else, we do our best to respond to new issues and pull requests. When
no approved contributor has been able to respond within 14 days of one being opened,
the bot closes it for now and leaves a note explaining why. This reflects our limited
capacity, not the value of your report or contribution, and you are welcome to ask
for it to be reopened.

Once an approved contributor leaves feedback on a pull request, the next step is
usually yours. If 14 days pass without a reply, the bot posts a friendly reminder.
If there is still no reply 7 days after that, it closes the pull request for now to
keep the review queue clean, and you are welcome to pick it up again whenever you
are ready. A reply does not need to be much: a comment, a reply to a review comment,
a review, or re-requesting review all count, and a short "still working on this" is
plenty. Pushing commits on its own does not count, because we cannot tell from
commits whether the feedback has been addressed. New feedback starts the clock again.

If you need more time, that is fine. On a reviewed pull request, just say so in a
comment; that counts as a reply. Otherwise, ask in a comment and an approved
contributor can add the `keep-open` label, which tells the bot to leave it alone for
as long as the label stays on. If it was already closed, they can reopen it instead.
