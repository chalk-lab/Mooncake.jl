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
