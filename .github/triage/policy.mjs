// Pure policy evaluation: the clock, configuration and GitHub history are supplied by the caller.
const DAY = 24 * 60 * 60 * 1000;
export const DEFAULTS = {
  botLogin: 'github-actions[bot]', overrideLabel: 'keep-open',
  engageDays: 14, replyDays: 14, graceDays: 7,
};

export const marker = (kind, key) => `<!-- mooncake-triage:v1:${kind}:${key} -->`;
export const isHuman = (user) => user?.type === 'User' && !user.login.endsWith('[bot]');
export const isBot = (user, botLogin) => user?.type === 'Bot' && user.login === botLogin;

export function timestamp(value) {
  const result = Date.parse(value);
  if (!Number.isFinite(result)) throw new Error(`Invalid timestamp: ${value}`);
  return result;
}

export function parseApproved(text) {
  const names = text.split('\n').map((line) => line.split('#')[0].trim()).filter(Boolean);
  if (names.some((name) => !/^[a-z\d][a-z\d-]*$/i.test(name))) {
    throw new Error('The approved-contributor list must contain GitHub usernames, one per line.');
  }
  return new Set(names.map((name) => name.toLowerCase()));
}

export function evaluate({ item, timeline = [], reviews = [], reviewComments = [] }, options) {
  const {
    approved, now, since, botLogin, overrideLabel, engageDays, replyDays, graceDays,
  } = { ...DEFAULTS, ...options };
  const isContributor = (user) => isHuman(user) && approved.has(user.login.toLowerCase());
  const keep = (reason) => ({ action: 'keep', reason });
  if (item.state !== 'open') return keep('Already closed');
  if (timestamp(item.created_at) < since) return keep('Before rollout date');
  if (!isHuman(item.user)) return keep('Non-human author');
  if (isContributor(item.user)) return keep('Contributor author');
  if (timeline.some((event) => isContributor(event.actor) && (
    event.event === 'reopened' || (event.event === 'labeled' && event.label.name === overrideLabel)
  ))) return keep('Contributor override');

  const comments = timeline.filter((event) => event.event === 'commented');
  const unengaged = () => {
    const due = timestamp(item.created_at) + engageDays * DAY;
    return now < due ? keep(`Within initial ${engageDays} days`) : {
      action: 'close', key: 'unengaged', reason: `No contributor engagement in ${engageDays} days`,
    };
  };

  if (!item.pull_request) {
    const triaged = comments.some((comment) => isContributor(comment.user)) || timeline.some(
      (event) => event.event === 'labeled' && isContributor(event.actor),
    );
    return triaged ? keep('Contributor triaged issue') : unengaged();
  }

  const reviewById = new Map(reviews.map((review) => [review.id, review]));
  const interactions = comments.map((comment) => ({
    key: `comment:${comment.id}`, user: comment.user,
    at: timestamp(comment.created_at), url: comment.html_url,
  }));
  for (const review of reviews) {
    if (review.state === 'PENDING' || !review.submitted_at) continue;
    interactions.push({
      key: `review:${review.id}`, user: review.user,
      at: timestamp(review.submitted_at), url: review.html_url,
    });
  }
  for (const comment of reviewComments) {
    const review = reviewById.get(comment.pull_request_review_id);
    if (review?.state === 'PENDING') continue;
    // Draft inline comments become visible only when their review is submitted.
    interactions.push({
      key: `inline:${comment.id}`, user: comment.user, url: comment.html_url,
      at: Math.max(timestamp(comment.created_at), review?.submitted_at ? timestamp(review.submitted_at) : 0),
    });
  }
  const feedback = interactions.filter((entry) => isContributor(entry.user)).sort(
    (a, b) => b.at - a.at || a.key.localeCompare(b.key),
  )[0];
  if (!feedback) return unengaged();

  const byAuthor = (user) => isHuman(user) && user.id === item.user.id;
  // Same-second replies count: GitHub timestamps cannot reliably order them further.
  const responded = interactions.some((entry) => byAuthor(entry.user) && entry.at >= feedback.at)
    || timeline.some((event) => event.event === 'review_requested' && byAuthor(event.actor)
      && timestamp(event.created_at) >= feedback.at);
  if (responded) return keep('Author responded; awaiting contributors');
  if (now < feedback.at + replyDays * DAY) return keep('Within feedback response period');

  const reminder = comments.filter((comment) => isBot(comment.user, botLogin)
    && comment.body?.includes(marker('reminder', feedback.key))
    && timestamp(comment.created_at) >= feedback.at).sort(
    (a, b) => timestamp(a.created_at) - timestamp(b.created_at),
  )[0];
  if (!reminder) return {
    action: 'remind', key: feedback.key, feedbackUrl: feedback.url,
    reason: `Contributor feedback unanswered for ${replyDays} days`,
  };
  if (now < timestamp(reminder.created_at) + graceDays * DAY) return keep('Within reminder grace period');
  return { action: 'close', key: feedback.key, reason: `No author response ${graceDays} days after reminder` };
}
