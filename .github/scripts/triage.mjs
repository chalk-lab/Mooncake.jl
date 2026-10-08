import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const DAY = 24 * 60 * 60 * 1000;
const BOT = 'github-actions[bot]';
const marker = (kind, key) => `<!-- mooncake-triage:v1:${kind}:${key} -->`;
const isHuman = (user) => user?.type === 'User' && !user.login.endsWith('[bot]');
const isBot = (user) => user?.type === 'Bot' && user.login === BOT;

function timestamp(value) {
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

// Tests may scale a day; the production CLI always uses real 24-hour days.
// Pure policy evaluation: the clock and GitHub history are supplied by the caller.
export function evaluate({ item, timeline = [], reviews = [], reviewComments = [], maintainers = new Set() }, {
  approved, now, since, dayMs = DAY,
}) {
  const maintainer = (user) => isHuman(user) && maintainers.has(user.login.toLowerCase());
  const trusted = (user) => maintainer(user) || (isHuman(user) && approved.has(user.login.toLowerCase()));
  const keep = (reason) => ({ action: 'keep', reason });
  if (item.state !== 'open') return keep('Already closed');
  if (timestamp(item.created_at) < since) return keep('Before rollout date');
  if (!isHuman(item.user)) return keep('Non-human author');
  if (maintainer(item.user)) return keep('Maintainer author');
  if (trusted(item.user)) return keep('Approved author');
  if (timeline.some((event) => maintainer(event.actor) && (
    event.event === 'reopened' || (event.event === 'labeled' && event.label.name === 'keep-open')
  ))) return keep('Maintainer override');

  const comments = timeline.filter((event) => event.event === 'commented');
  const unengaged = () => {
    const due = timestamp(item.created_at) + 14 * dayMs;
    return now < due ? keep('Within initial 14 days') : {
      action: 'close', key: 'unengaged', reason: 'No contributor engagement in 14 days',
    };
  };

  if (!item.pull_request) {
    const triaged = comments.some((comment) => trusted(comment.user)) || timeline.some(
      (event) => event.event === 'labeled' && trusted(event.actor),
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
  const feedback = interactions.filter((entry) => trusted(entry.user)).sort(
    (a, b) => b.at - a.at || a.key.localeCompare(b.key),
  )[0];
  if (!feedback) return unengaged();

  const byAuthor = (user) => isHuman(user) && user.id === item.user.id;
  // Same-second replies count: GitHub timestamps cannot reliably order them further.
  const responded = interactions.some((entry) => byAuthor(entry.user) && entry.at >= feedback.at)
    || timeline.some((event) => event.event === 'review_requested' && byAuthor(event.actor)
      && timestamp(event.created_at) >= feedback.at);
  if (responded) return keep('Author responded; awaiting contributors');
  if (now < feedback.at + 14 * dayMs) return keep('Within feedback response period');

  const reminder = comments.filter((comment) => isBot(comment.user)
    && comment.body?.includes(marker('reminder', feedback.key))
    && timestamp(comment.created_at) >= feedback.at).sort(
    (a, b) => timestamp(a.created_at) - timestamp(b.created_at),
  )[0];
  if (!reminder) return {
    action: 'remind', key: feedback.key, feedbackUrl: feedback.url,
    reason: 'Contributor feedback unanswered for 14 days',
  };
  if (now < timestamp(reminder.created_at) + 7 * dayMs) return keep('Within reminder grace period');
  return { action: 'close', key: feedback.key, reason: 'No author response seven days after reminder' };
}

async function loadSnapshot(api, repo, number, permissions) {
  const base = `repos/${repo}`;
  const item = await api('GET', `${base}/issues/${number}`);
  const timeline = await api('GET', `${base}/issues/${number}/timeline?per_page=100`, { paginate: true });
  const reviews = item.pull_request
    ? await api('GET', `${base}/pulls/${number}/reviews?per_page=100`, { paginate: true }) : [];
  const reviewComments = item.pull_request
    ? await api('GET', `${base}/pulls/${number}/comments?per_page=100`, { paginate: true }) : [];
  const users = [item.user, ...timeline.flatMap((event) => [event.actor, event.user]),
    ...reviews.map((review) => review.user), ...reviewComments.map((comment) => comment.user)];
  const maintainers = new Set();
  for (const user of users.filter(isHuman)) {
    const login = user.login.toLowerCase();
    if (!permissions.has(login)) {
      const { permission } = await api('GET', `${base}/collaborators/${encodeURIComponent(login)}/permission`);
      // GitHub maps maintain/custom write roles to write, and triage roles to read.
      if (!['admin', 'write', 'read', 'none'].includes(permission)) {
        throw new Error(`Unknown repository permission for ${login}: ${permission}`);
      }
      permissions.set(login, permission);
    }
    if (['admin', 'write'].includes(permissions.get(login))) maintainers.add(login);
  }
  return { item, timeline, reviews, reviewComments, maintainers };
}

export async function runTriage({ api, repo, approved, since, now = Date.now(), apply = false, dayMs = DAY }) {
  const items = await api('GET', `repos/${repo}/issues?state=open&per_page=100`, { paginate: true });
  const results = [];
  const permissions = new Map(); // Cache within this scan only; refresh roles on the next run.
  const options = { approved, since, now, dayMs };
  for (const item of items) {
    const result = { number: item.number, kind: item.pull_request ? 'pr' : 'issue' };
    try {
      let snapshot = await loadSnapshot(api, repo, item.number, permissions);
      const decision = evaluate(snapshot, options);
      Object.assign(result, decision);
      if (apply && decision.action !== 'keep') {
        const unchanged = (current) => {
          const next = evaluate(current, options);
          return next.action === decision.action && next.key === decision.key;
        };
        snapshot = await loadSnapshot(api, repo, item.number, permissions);
        if (!unchanged(snapshot)) {
          results.push({ ...result, action: 'keep', reason: 'Activity changed before writing' });
          continue;
        }
        const commentsPath = `repos/${repo}/issues/${item.number}/comments`;
        const policy = `https://github.com/${repo}/blob/HEAD/.github/TRIAGE.md`;
        if (decision.action === 'remind') {
          const feedbackUrl = decision.feedbackUrl || `https://github.com/${repo}/pull/${item.number}`;
          await api('POST', commentsPath, { body: {
            body: `@${item.user.login}, we have not seen your response to [this feedback](${feedbackUrl}) `
              + 'for 14 days. Please reply or re-request review within seven days of this reminder '
              + 'to keep this PR open. Commits alone do not count as a response.\n\n'
              + `[Triage policy](${policy}).\n\n${marker('reminder', decision.key)}`,
          } });
        } else {
          const notice = marker('close', decision.key);
          const reopenedAt = Math.max(0, ...snapshot.timeline
            .filter((event) => event.event === 'reopened').map((event) => timestamp(event.created_at)));
          if (!snapshot.timeline.some((event) => event.event === 'commented'
            && isBot(event.user) && event.body?.includes(notice)
            && timestamp(event.created_at) >= reopenedAt)) {
            await api('POST', commentsPath, { body: {
              body: `Closing automatically under the [triage policy](${policy}): ${decision.reason.toLowerCase()}. `
                + 'This is not a judgment about the report or fix. For reconsideration, reply here and '
                + 'ask a maintainer to reopen it or add `keep-open`.\n\n' + notice,
            } });
          }
          // Recheck after posting as well: a reply or override can arrive during the write.
          if (!unchanged(await loadSnapshot(api, repo, item.number, permissions))) {
            results.push({ ...result, action: 'keep', reason: 'Activity changed before closing' });
            continue;
          }
          const path = `repos/${repo}/${item.pull_request ? 'pulls' : 'issues'}/${item.number}`;
          await api('PATCH', path, { body: item.pull_request
            ? { state: 'closed' } : { state: 'closed', state_reason: 'not_planned' } });
        }
        result.applied = true;
      }
    } catch (error) {
      Object.assign(result, { action: 'error', reason: error.message });
    }
    results.push(result);
  }
  return results;
}

// gh handles authentication and pagination. Guard writes at the transport boundary too.
export function githubApi({ allowWrites = false, exec = execFileSync } = {}) {
  return async (method, path, { paginate = false, body = {} } = {}) => {
    if (method !== 'GET' && !allowWrites) throw new Error('Writes are disabled');
    const args = ['api', '--method', method, path];
    if (paginate) args.push('--paginate', '--slurp');
    for (const [key, value] of Object.entries(body)) args.push('-f', `${key}=${value}`);
    const output = exec('gh', args, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
    const data = output.trim() ? JSON.parse(output) : null;
    return paginate ? data.flat() : data;
  };
}

export function configuration(args, env) {
  const { values } = parseArgs({ args, options: {
    repo: { type: 'string', default: env.GITHUB_REPOSITORY },
    since: { type: 'string' }, apply: { type: 'boolean', default: false },
  } });
  if (!/^[\w.-]+\/[\w.-]+$/.test(values.repo || '')) throw new Error('Supply --repo OWNER/REPO');
  if (values.apply && (!values.since || env.GITHUB_ACTIONS !== 'true')) {
    throw new Error('--apply requires an explicit --since date and GitHub Actions with GITHUB_TOKEN');
  }
  const date = values.since || '1970-01-01';
  const since = timestamp(date);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || new Date(since).toISOString().slice(0, 10) !== date) {
    throw new Error('--since must be a valid YYYY-MM-DD date (UTC)');
  }
  return { repo: values.repo, since, apply: values.apply };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const config = configuration(process.argv.slice(2), process.env);
    const approved = parseApproved(readFileSync(new URL('../APPROVED_CONTRIBUTORS', import.meta.url), 'utf8'));
    const results = await runTriage({ ...config, approved, api: githubApi({ allowWrites: config.apply }) });
    console.log(JSON.stringify({
      mode: config.apply ? 'apply' : 'dry-run', since: new Date(config.since).toISOString(), results,
    }, null, 2));
    if (results.some((result) => result.action === 'error')) process.exitCode = 1;
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
