import { execFileSync } from 'node:child_process';
import { DEFAULTS, evaluate, isBot, marker, timestamp } from './policy.mjs';

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
      let snapshot = await loadSnapshot(api, repo, item.number);
      const decision = evaluate(snapshot, options);
      Object.assign(result, decision);
      if (apply && decision.action !== 'keep') {
        const unchanged = (current) => {
          const next = evaluate(current, options);
          return next.action === decision.action && next.key === decision.key;
        };
        snapshot = await loadSnapshot(api, repo, item.number);
        if (!unchanged(snapshot)) {
          results.push({ ...result, action: 'keep', reason: 'Activity changed before writing' });
          continue;
        }
        const commentsPath = `repos/${repo}/issues/${item.number}/comments`;
        const policy = `https://github.com/${repo}/blob/HEAD/.github/triage/README.md`;
        if (decision.action === 'remind') {
          const feedbackUrl = decision.feedbackUrl || `https://github.com/${repo}/pull/${item.number}`;
          await api('POST', commentsPath, { body: {
            body: `@${item.user.login}, we have not seen your response to [this feedback](${feedbackUrl}) `
              + `for ${options.replyDays} days. Please reply or re-request review within `
              + `${options.graceDays} days of this reminder to keep this PR open. `
              + 'Commits alone do not count as a response.\n\n'
              + `[Triage policy](${policy}).\n\n${marker('reminder', decision.key)}`,
          } });
        } else {
          const notice = marker('close', decision.key);
          const reopenedAt = Math.max(0, ...snapshot.timeline
            .filter((event) => event.event === 'reopened').map((event) => timestamp(event.created_at)));
          if (!snapshot.timeline.some((event) => event.event === 'commented'
            && isBot(event.user, options.botLogin) && event.body?.includes(notice)
            && timestamp(event.created_at) >= reopenedAt)) {
            await api('POST', commentsPath, { body: {
              body: `Closing automatically under the [triage policy](${policy}): ${decision.reason.toLowerCase()}. `
                + 'This is not a judgment about the report or fix. For reconsideration, reply here and '
                + `ask a listed contributor to reopen it or add \`${options.overrideLabel}\`.\n\n` + notice,
            } });
          }
          // Recheck after posting as well: a reply or override can arrive during the write.
          if (!unchanged(await loadSnapshot(api, repo, item.number))) {
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
