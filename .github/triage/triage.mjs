// Contributor triage: closes stalled issues and PRs per README.md in this directory.
//   node --test .github/triage/triage.test.mjs                                 # tests, no credentials
//   GITHUB_REPOSITORY=chalk-lab/Mooncake.jl node .github/triage/triage.mjs     # read-only preview via gh
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const DAY = 24 * 60 * 60 * 1000;
const DEFAULTS = {
  botLogin: 'github-actions[bot]', overrideLabel: 'keep-open',
  engageDays: 14, replyDays: 14, graceDays: 7,
};

const marker = (kind, key) => `<!-- mooncake-triage:v1:${kind}:${key} -->`;
const isHuman = (user) => user?.type === 'User' && !user.login.endsWith('[bot]');
const isBot = (user, botLogin) => user?.type === 'Bot' && user.login === botLogin;

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

// Pure policy evaluation: the clock, configuration and GitHub history are supplied by the caller.
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
    || timeline.some((event) => ['review_requested', 'reopened'].includes(event.event) && byAuthor(event.actor)
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
  return {
    action: 'close', key: feedback.key, feedbackUrl: feedback.url,
    reason: `No author response ${graceDays} days after reminder`,
  };
}

async function loadSnapshot(api, repo, number) {
  const base = `repos/${repo}`;
  const item = await api('GET', `${base}/issues/${number}`);
  const timeline = await api('GET', `${base}/issues/${number}/timeline?per_page=100`, { paginate: true });
  const reviews = item.pull_request
    ? await api('GET', `${base}/pulls/${number}/reviews?per_page=100`, { paginate: true }) : [];
  const reviewComments = item.pull_request
    ? await api('GET', `${base}/pulls/${number}/comments?per_page=100`, { paginate: true }) : [];
  return { item, timeline, reviews, reviewComments };
}

// `options` are policy options (see DEFAULTS) plus `approved`, `since` and `now`.
export async function runTriage({ api, repo, apply = false, now = Date.now(), ...rest }) {
  const options = { ...DEFAULTS, now, ...rest };
  const items = await api('GET', `repos/${repo}/issues?state=open&per_page=100`, { paginate: true });
  const results = [];
  for (const item of items) {
    const result = { number: item.number, kind: item.pull_request ? 'pr' : 'issue' };
    try {
      const snapshot = await loadSnapshot(api, repo, item.number);
      const decision = evaluate(snapshot, options);
      Object.assign(result, decision);
      if (apply && decision.action !== 'keep') {
        const commentsPath = `repos/${repo}/issues/${item.number}/comments`;
        const post = (text, notice) => api('POST', commentsPath, { body: { body: `${text}\n\n<sub>Posted `
          + `automatically under the [triage policy](https://github.com/${repo}/blob/HEAD/.github/triage/README.md).`
          + `</sub>\n\n${notice}` } });
        const feedback = `[review feedback](${decision.feedbackUrl})`;
        if (decision.action === 'remind') {
          await post(`Hi @${item.user.login}, a friendly reminder that there is ${feedback} on this PR `
            + 'waiting for your reply. When you get a chance, please leave a comment (even a short '
            + '"still working on it" is enough) or re-request review once it is ready; we cannot tell '
            + 'from commits alone whether feedback has been addressed. If there is no reply within '
            + `${options.graceDays} days, this PR will be closed to keep the review queue manageable. `
            + 'Nothing is lost if that happens, and it can be reopened at any time.',
          marker('reminder', decision.key));
        } else {
          const notice = marker('close', decision.key);
          const reopenedAt = Math.max(0, ...snapshot.timeline
            .filter((event) => event.event === 'reopened').map((event) => timestamp(event.created_at)));
          if (!snapshot.timeline.some((event) => event.event === 'commented'
            && isBot(event.user, options.botLogin) && event.body?.includes(notice)
            && timestamp(event.created_at) >= reopenedAt)) {
            const why = decision.key === 'unengaged'
              ? `we have not been able to respond within ${options.engageDays} days. That reflects our `
                + 'limited capacity, not a judgment on the report or fix'
              : `the ${feedback} has not had a reply for a while. This is not a judgment on the work`;
            await post(`Closing this for now, as ${why}. Nothing is lost: the discussion stays here. `
              + 'If it is still relevant to you, leave a comment and a maintainer can reopen it.', notice);
          }
          // Recheck after posting: a reply or override can arrive during the write.
          const next = evaluate(await loadSnapshot(api, repo, item.number), options);
          if (next.action !== decision.action || next.key !== decision.key) {
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

// Configuration comes from the environment so the workflow can pass its variables
// straight through: GITHUB_REPOSITORY, TRIAGE_START_DATE and TRIAGE_APPLY.
export function configuration(env) {
  const repo = env.GITHUB_REPOSITORY;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo || '')) throw new Error('Set GITHUB_REPOSITORY=OWNER/REPO');
  const date = env.TRIAGE_START_DATE || '';
  const apply = env.TRIAGE_APPLY === 'true';
  if (apply && !date) throw new Error('TRIAGE_APPLY=true requires TRIAGE_START_DATE');
  if (date && !(/^\d{4}-\d{2}-\d{2}$/.test(date) && new Date(Date.parse(date)).toISOString().startsWith(date))) {
    throw new Error('TRIAGE_START_DATE must be a valid YYYY-MM-DD date (UTC)');
  }
  return { repo, since: date ? timestamp(date) : 0, apply };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const config = configuration(process.env);
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
