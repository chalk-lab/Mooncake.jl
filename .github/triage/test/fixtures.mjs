import assert from 'node:assert/strict';
import { runTriage } from '../github.mjs';
import { evaluate } from '../policy.mjs';

export const author = { login: 'outside-author', id: 101, type: 'User' };
export const contributor = { login: 'Approved-Contributor', id: 202, type: 'User' };
export const other = { login: 'other-person', id: 303, type: 'User' };
export const bot = { login: 'github-actions[bot]', id: 404, type: 'Bot' };
export const approved = new Set(['approved-contributor']);
export const at = (date) => Date.parse(date);
export const item = (pr = false) => ({
  number: 7, state: 'open', user: author, author_association: 'CONTRIBUTOR',
  created_at: '2026-01-01T12:00:00Z', updated_at: '2026-02-01T11:59:59Z',
  ...(pr ? { pull_request: {} } : {}),
});
export const comment = (id, user, created_at, body = '') => ({
  event: 'commented', id, user, actor: user, created_at, body,
  html_url: `https://github.com/example/repo/pull/7#issuecomment-${id}`,
});
export const feedback = comment(11, contributor, '2026-01-03T12:00:00Z');
export const reminder = comment(22, bot, '2026-01-25T12:00:00Z',
  '<!-- mooncake-triage:v1:reminder:comment:11 -->');
export const snapshot = (pr = true) => ({ item: item(pr), timeline: [], reviews: [], reviewComments: [] });
export const decide = (data, now = '2026-02-01T12:00:00Z', since = '2026-01-01', options = {}) => evaluate(data, {
  approved, now: at(now), since: at(since), ...options,
});

// Fake the wire API, not the policy. Recorded writes are checked separately below.
export function fixture(data, now = '2026-02-01T12:00:00Z') {
  const calls = [];
  let onCall = () => {};
  const api = async (method, path, options = {}) => {
    calls.push({ method, path, ...options });
    onCall(method, path);
    if (method === 'GET') {
      if (path.includes('?')) assert.equal(options.paginate, true);
      if (path.includes('/issues?')) return data.item.state === 'open' ? [structuredClone(data.item)] : [];
      if (path.includes('/timeline?')) return structuredClone(data.timeline);
      if (path.includes('/reviews?')) return structuredClone(data.reviews);
      if (path.includes('/pulls/7/comments?')) return structuredClone(data.reviewComments);
      assert.equal(path, 'repos/example/repo/issues/7');
      return structuredClone(data.item);
    }
    if (method === 'POST') {
      assert.equal(path, 'repos/example/repo/issues/7/comments');
      const posted = comment(1000 + calls.length, bot, now, options.body.body);
      data.timeline.push(posted);
      return posted;
    }
    assert.equal(method, 'PATCH');
    data.item.state = options.body.state;
    return structuredClone(data.item);
  };
  return {
    calls, data, api,
    hook: (callback) => { onCall = callback; },
    run: (apply = true) => runTriage({ api, repo: 'example/repo', approved, since: at('2026-01-01'), now: at(now), apply }),
    writes: () => calls.filter((call) => call.method !== 'GET'),
  };
}
