import assert from 'node:assert/strict';
import test from 'node:test';
import { githubApi, runTriage } from '../github.mjs';
import {
  approved, at, author, bot, comment, contributor, feedback, fixture, item, reminder, snapshot,
} from './fixtures.mjs';

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
  assert.match(f.writes()[0].body.body, /7 days of this reminder/);
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

test('sandbox day scaling reaches the writer without writing in dry-run', async () => {
  const data = snapshot();
  data.timeline = [comment(11, contributor, '2026-01-01T12:00:00Z'),
    { ...reminder, created_at: '2026-01-01T12:16:00Z' }];
  const f = fixture(data);
  const result = await runTriage({
    api: f.api, repo: 'example/repo', approved, since: 0,
    now: at('2026-01-01T12:23:00Z'), dayMs: 60_000,
  });
  assert.equal(result[0].action, 'close');
});
