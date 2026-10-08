import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { configuration, evaluate, githubApi, parseApproved, runTriage } from './triage.mjs';

const author = { login: 'outside-author', id: 101, type: 'User' };
const contributor = { login: 'Approved-Contributor', id: 202, type: 'User' };
const other = { login: 'other-person', id: 303, type: 'User' };
const bot = { login: 'github-actions[bot]', id: 404, type: 'Bot' };
const approved = new Set(['approved-contributor']);
const at = (date) => Date.parse(date);
const item = (pr = false) => ({
  number: 7, state: 'open', user: author, author_association: 'CONTRIBUTOR',
  created_at: '2026-01-01T12:00:00Z', updated_at: '2026-02-01T11:59:59Z',
  ...(pr ? { pull_request: {} } : {}),
});
const comment = (id, user, created_at, body = '') => ({
  event: 'commented', id, user, actor: user, created_at, body,
  html_url: `https://github.com/example/repo/pull/7#issuecomment-${id}`,
});
const feedback = comment(11, contributor, '2026-01-03T12:00:00Z');
const reminder = comment(22, bot, '2026-01-25T12:00:00Z',
  '<!-- mooncake-triage:v1:reminder:comment:11 -->');
const snapshot = (pr = true) => ({ item: item(pr), timeline: [], reviews: [], reviewComments: [] });
const decide = (data, now = '2026-02-01T12:00:00Z', since = '2026-01-01') => evaluate(data, {
  approved, now: at(now), since: at(since),
});

test('allowlist accepts empty input, validates names, and alone determines contributor status', () => {
  assert.deepEqual(parseApproved('# approved\nApproved-Contributor # name\n\n'), approved);
  for (const text of ['', ' \n\t', '# empty\n']) {
    assert.deepEqual(parseApproved(text), new Set());
  }
  assert.throws(() => parseApproved('two names'), /usernames/);
  parseApproved(readFileSync(new URL('../APPROVED_CONTRIBUTORS', import.meta.url), 'utf8'));
  assert.equal(decide({ item: { ...item(), user: contributor } }).reason, 'Contributor author');
  assert.equal(evaluate({ item: { ...item(), user: contributor } }, {
    approved: parseApproved(''), now: at('2026-02-01T12:00:00Z'), since: 0,
  }).action, 'close');
  for (const author_association of ['OWNER', 'MEMBER', 'CONTRIBUTOR']) {
    assert.equal(decide({ item: { ...item(), author_association } }).action, 'close');
  }
});

for (const pr of [false, true]) {
  test(`${pr ? 'PR' : 'issue'} closes at 14 days, not earlier; recent activity is not the clock`, () => {
    const data = snapshot(pr);
    data.timeline = [comment(1, author, '2026-01-14T12:00:00Z'),
      comment(2, bot, '2026-01-14T13:00:00Z'), comment(3, other, '2026-01-14T14:00:00Z')];
    assert.equal(decide(data, '2026-01-15T11:59:59.999Z').action, 'keep');
    assert.equal(decide(data, '2026-01-15T12:00:00Z').action, 'close');
    assert.equal(decide(data, undefined, '2026-01-02').reason, 'Before rollout date');
    data.item.state = 'closed';
    assert.equal(decide(data).reason, 'Already closed');
  });
}

test('issue exemptions use label/comment actors, not current labels or bot activity', () => {
  const data = snapshot(false);
  data.item.labels = [{ name: 'bug' }];
  for (const actor of [author, other, bot]) {
    data.timeline = [{ event: 'labeled', actor, label: { name: 'bug' } }];
    assert.equal(decide(data).action, 'close');
  }
  data.timeline = [{ event: 'labeled', actor: contributor, label: { name: 'bug' } },
    { event: 'unlabeled', actor: contributor, label: { name: 'bug' } }];
  data.item.labels = [];
  assert.equal(decide(data).reason, 'Contributor triaged issue');
  data.timeline = [feedback];
  assert.equal(decide(data).reason, 'Contributor triaged issue');
});

test('only listed contributors can exempt PRs with keep-open or reopening', () => {
  const data = snapshot();
  for (const event of ['reopened', 'labeled']) {
    for (const actor of [author, other, bot, contributor]) {
      data.timeline = [{ event, actor, label: { name: 'keep-open' } }];
      assert.equal(decide(data).action, actor === contributor ? 'keep' : 'close');
    }
  }
  data.timeline = [{ event: 'labeled', actor: contributor, label: { name: 'bug' } }];
  assert.equal(decide(data).action, 'close');
});

for (const kind of ['comment', 'inline', 'COMMENTED', 'CHANGES_REQUESTED', 'APPROVED', 'DISMISSED']) {
  test(`${kind} counts as feedback; remind exactly 14 days after feedback, not opening`, () => {
    const data = snapshot();
    if (kind === 'comment') data.timeline = [feedback];
    else if (kind === 'inline') data.reviewComments = [feedback];
    else data.reviews = [{ id: 42, user: contributor, state: kind, submitted_at: feedback.created_at }];
    assert.equal(decide(data, '2026-01-17T11:59:59.999Z').action, 'keep');
    assert.equal(decide(data, '2026-01-17T12:00:00Z').action, 'remind');
  });
}

test('pending reviews do not count, and drafted inline comments start their clock at publication', () => {
  const data = snapshot();
  data.reviews = [{ id: 8, state: 'PENDING', user: contributor, submitted_at: null }];
  data.reviewComments = [{ ...feedback, pull_request_review_id: 8 }];
  assert.equal(decide(data, '2026-01-20T12:00:00Z').action, 'close');
  data.reviews[0] = { ...data.reviews[0], state: 'COMMENTED', submitted_at: '2026-01-18T12:00:00Z' };
  assert.equal(decide(data, '2026-01-20T12:00:00Z').action, 'keep');
});

test('only the author can respond; inline replies, reviews and re-requested review count', () => {
  const data = snapshot();
  data.timeline = [feedback];
  for (const user of [other, bot, author]) {
    data.reviewComments = [comment(55, user, '2026-01-10T12:00:00Z')];
    assert.equal(decide(data).action, user === author ? 'keep' : 'remind');
  }
  data.reviewComments = [];
  data.reviews = [{ id: 55, user: author, state: 'COMMENTED', submitted_at: '2026-01-10T12:00:00Z' }];
  assert.equal(decide(data).reason, 'Author responded; awaiting contributors');
  data.reviews = [];
  data.timeline.push({ event: 'review_requested', actor: author, created_at: '2026-01-11T12:00:00Z' });
  assert.equal(decide(data).reason, 'Author responded; awaiting contributors');
  data.timeline[1].actor = other;
  assert.equal(decide(data).action, 'remind');
  data.timeline[1] = { event: 'committed', actor: author, created_at: '2026-01-11T12:00:00Z' };
  assert.equal(decide(data).action, 'remind');
});

test('author must respond after the latest feedback, irrespective of API ordering or comment edits', () => {
  const data = snapshot();
  const newer = comment(9, contributor, '2026-01-20T12:00:00Z');
  data.timeline = [newer, feedback, comment(18, author, '2026-01-19T12:00:00Z')];
  data.timeline[2].updated_at = '2026-02-10T12:00:00Z';
  assert.equal(decide(data).action, 'keep'); // New feedback has its own 14 days.
  assert.equal(decide(data, '2026-02-03T12:00:00Z').key, 'comment:9');
  data.timeline.push(comment(99, author, newer.created_at));
  assert.equal(decide(data, '2026-02-03T12:00:00Z').reason, 'Author responded; awaiting contributors');
});

test('delayed reminders grant a full seven days; spoofed or superseded reminders cannot close a PR', () => {
  const data = snapshot();
  data.timeline = [feedback, reminder];
  assert.equal(decide(data, '2026-02-01T11:59:59.999Z').action, 'keep');
  assert.equal(decide(data, '2026-02-01T12:00:00Z').action, 'close');
  for (const user of [other, { ...bot, login: 'other[bot]' }, { ...bot, type: 'User' }]) {
    data.timeline[1] = { ...reminder, user };
    assert.equal(decide(data).action, 'remind');
  }
  data.timeline[1] = reminder;
  data.timeline.push(comment(44, contributor, '2026-01-30T12:00:00Z'));
  assert.equal(decide(data).action, 'keep');
  assert.equal(decide(data, '2026-02-13T12:00:00Z').action, 'remind');
  data.timeline.push(comment(45, author, '2026-02-14T12:00:00Z'));
  assert.equal(decide(data, '2026-02-15T12:00:00Z').action, 'keep');
});

// Fake the wire API, not the policy. Recorded writes are checked separately below.
function fixture(data, now = '2026-02-01T12:00:00Z') {
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

test('non-human authors never receive inactivity reminders or closures', async () => {
  for (const pr of [false, true]) {
    for (const timeline of [[], [feedback], [feedback, reminder]]) {
      const f = fixture({ ...snapshot(pr), item: { ...item(pr), user: bot }, timeline });
      assert.equal((await f.run())[0].reason, 'Non-human author');
      assert.deepEqual(f.writes(), []);
    }
  }
});

test('scans use the current contributor list without permission lookups', async () => {
  const f = fixture({ ...snapshot(), timeline: [{ event: 'reopened', actor: contributor }] });
  assert.equal((await f.run(false))[0].reason, 'Contributor override');
  const [result] = await runTriage({
    api: f.api, repo: 'example/repo', approved: new Set(), since: 0, now: at('2026-02-01'),
  });
  assert.equal(result.action, 'close');
  assert.ok(f.calls.every((call) => !call.path.includes('/collaborators/')));
  assert.deepEqual(f.writes(), []);
});

test('dry-run produces the same decision as apply without posting or patching anything', async () => {
  const f = fixture({ ...snapshot(), timeline: [feedback] });
  const [preview] = await f.run(false);
  assert.equal(preview.action, 'remind');
  assert.deepEqual(f.writes(), []);
  const [applied] = await f.run();
  assert.equal(applied.action, preview.action);
  assert.equal(applied.applied, true);
  assert.match(f.writes()[0].body.body, /seven days of this reminder/);
  assert.match(f.writes()[0].body.body, /<!-- mooncake-triage:v1:reminder:comment:11 -->/);
  assert.equal((await f.run())[0].action, 'keep');
  assert.equal(f.writes().length, 1); // No duplicate reminder on a rerun.
});

for (const pr of [false, true]) {
  test(`closes ${pr ? 'PR' : 'issue'} via the correct endpoint and does not repeat writes`, async () => {
    const f = fixture(snapshot(pr));
    assert.equal((await f.run())[0].applied, true);
    assert.equal(f.writes()[0].method, 'POST');
    assert.deepEqual(f.writes()[1], {
      method: 'PATCH', path: `repos/example/repo/${pr ? 'pulls' : 'issues'}/7`,
      body: pr ? { state: 'closed' } : { state: 'closed', state_reason: 'not_planned' },
    });
    assert.deepEqual(await f.run(), []);
    assert.equal(f.writes().length, 2);
  });
}

test('re-closing after an author reopen posts a fresh notice, but a failed close retry does not', async () => {
  for (const pr of [false, true]) {
    const notice = `<!-- mooncake-triage:v1:close:${pr ? 'comment:11' : 'unengaged'} -->`;
    const f = fixture({ ...snapshot(pr), timeline: [
      ...(pr ? [feedback, reminder] : []),
      comment(30, bot, '2026-01-31T12:00:00Z', notice),
      { event: 'reopened', actor: author, created_at: '2026-02-01T11:00:00Z' },
    ] });
    f.hook((method) => { if (method === 'PATCH') throw new Error('Temporary close failure'); });
    assert.equal((await f.run())[0].action, 'error');
    assert.equal(f.writes().filter((call) => call.method === 'POST').length, 1);
    assert.ok(f.writes()[0].body.body.includes(notice));
    f.hook(() => {});
    assert.equal((await f.run())[0].applied, true);
    assert.equal(f.data.item.state, 'closed');
    assert.equal(f.writes().filter((call) => call.method === 'POST').length, 1);
  }
});

test('rechecks before writes and before closure; late responses and overrides win', async () => {
  const f = fixture({ ...snapshot(), timeline: [feedback, reminder] });
  let reads = 0;
  f.hook((method, path) => {
    if (method === 'GET' && path.endsWith('/issues/7') && ++reads === 2) {
      f.data.timeline.push(comment(88, author, '2026-01-31T12:00:00Z'));
    }
  });
  assert.equal((await f.run())[0].reason, 'Activity changed before writing');
  assert.deepEqual(f.writes(), []);

  const g = fixture({ ...snapshot(), timeline: [feedback, reminder] });
  g.hook((method) => {
    if (method === 'POST') g.data.timeline.push({ event: 'reopened', actor: contributor });
  });
  assert.equal((await g.run())[0].reason, 'Activity changed before closing');
  assert.equal(g.data.item.state, 'open');
  assert.equal(g.writes().filter((call) => call.method === 'PATCH').length, 0);
});

test('failed history reads never produce writes; partial writes are recoverable without duplicate comments', async () => {
  const f = fixture(snapshot());
  f.hook((method, path) => {
    if (path.includes('/reviews?')) throw new Error('API unavailable');
  });
  assert.equal((await f.run())[0].action, 'error');
  assert.deepEqual(f.writes(), []);

  const g = fixture(snapshot());
  g.hook((method) => { if (method === 'PATCH') throw new Error('API unavailable'); });
  assert.equal((await g.run())[0].action, 'error');
  assert.equal(g.data.item.state, 'open');
  g.hook(() => {});
  assert.equal((await g.run())[0].applied, true);
  assert.equal(g.writes().filter((call) => call.method === 'POST').length, 1);
});

test('transport flattens every page, uses explicit GET, and rejects writes by default', async () => {
  const api = githubApi({ exec: (command, args) => {
    assert.equal(command, 'gh');
    assert.deepEqual(args, ['api', '--method', 'GET', 'example/path', '--paginate', '--slurp']);
    return '[[{"id":1}],[{"id":2}]]';
  } });
  assert.deepEqual(await api('GET', 'example/path', { paginate: true }), [{ id: 1 }, { id: 2 }]);
  await assert.rejects(api('POST', 'example/path'), /Writes are disabled/);
  const failing = githubApi({ exec: () => { throw new Error('page 2 failed'); } });
  await assert.rejects(failing('GET', 'example/path', { paginate: true }), /page 2 failed/);
});

test('sandbox day scaling preserves 14:7 timing and is not configurable through the CLI', async () => {
  const data = snapshot();
  data.timeline = [comment(11, contributor, '2026-01-01T12:00:00Z')];
  const scaled = (time) => evaluate(data, { approved, since: 0, now: at(time), dayMs: 60_000 });
  assert.equal(scaled('2026-01-01T12:13:59.999Z').action, 'keep');
  assert.equal(scaled('2026-01-01T12:14:00Z').action, 'remind');
  data.timeline.push({ ...reminder, created_at: '2026-01-01T12:16:00Z' });
  assert.equal(scaled('2026-01-01T12:22:59.999Z').action, 'keep');
  assert.equal(scaled('2026-01-01T12:23:00Z').action, 'close');
  assert.equal(decide(data, '2026-01-01T12:23:00Z').action, 'keep'); // Production still uses days.

  const f = fixture(data);
  const result = await runTriage({
    api: f.api, repo: 'example/repo', approved, since: 0,
    now: at('2026-01-01T12:23:00Z'), dayMs: 60_000,
  });
  assert.equal(result[0].action, 'close');
  assert.deepEqual(f.writes(), []);
  assert.throws(() => configuration(['--repo', 'example/repo', '--day-ms', '60000'], {}), /Unknown option/);
});

test('CLI defaults to dry-run and rejects invalid dates or accidental local writes', () => {
  assert.deepEqual(configuration(['--repo', 'example/repo'], {}), {
    repo: 'example/repo', since: 0, apply: false,
  });
  for (const args of [[], ['--since', '2026-01-01']]) {
    assert.throws(() => configuration(['--repo', 'example/repo', '--apply', ...args], {}), /--apply requires/);
  }
  assert.throws(() => configuration(['--repo', 'example/repo', '--apply'], { GITHUB_ACTIONS: 'true' }), /explicit --since/);
  for (const date of ['2026-02-30', '2026-1-1', 'yesterday']) {
    assert.throws(() => configuration(['--repo', 'example/repo', '--since', date], {}));
  }
  assert.equal(configuration(['--since', '2026-01-01', '--apply'], {
    GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'example/repo',
  }).apply, true);
});
