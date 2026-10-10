import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { evaluate, parseApproved } from '../policy.mjs';
import {
  approved, at, author, bot, comment, contributor, decide, feedback, item, other, reminder, snapshot,
} from './fixtures.mjs';

test('allowlist accepts empty input, validates names, and alone determines contributor status', () => {
  assert.deepEqual(parseApproved('# approved\nApproved-Contributor # name\n\n'), approved);
  for (const text of ['', ' \n\t', '# empty\n']) {
    assert.deepEqual(parseApproved(text), new Set());
  }
  assert.throws(() => parseApproved('two names'), /usernames/);
  parseApproved(readFileSync(new URL('../contributors.txt', import.meta.url), 'utf8'));
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
  assert.equal(decide(data, undefined, undefined, { overrideLabel: 'bug' }).reason, 'Contributor override');
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
  assert.equal(decide(data, undefined, undefined, { botLogin: 'triage-app[bot]' }).action, 'remind');
  data.timeline.push(comment(44, contributor, '2026-01-30T12:00:00Z'));
  assert.equal(decide(data).action, 'keep');
  assert.equal(decide(data, '2026-02-13T12:00:00Z').action, 'remind');
  data.timeline.push(comment(45, author, '2026-02-14T12:00:00Z'));
  assert.equal(decide(data, '2026-02-15T12:00:00Z').action, 'keep');
});

test('sandbox day scaling preserves 14:7 timing', async () => {
  const data = snapshot();
  data.timeline = [comment(11, contributor, '2026-01-01T12:00:00Z')];
  const scaled = (time) => evaluate(data, { approved, since: 0, now: at(time), dayMs: 60_000 });
  assert.equal(scaled('2026-01-01T12:13:59.999Z').action, 'keep');
  assert.equal(scaled('2026-01-01T12:14:00Z').action, 'remind');
  data.timeline.push({ ...reminder, created_at: '2026-01-01T12:16:00Z' });
  assert.equal(scaled('2026-01-01T12:22:59.999Z').action, 'keep');
  assert.equal(scaled('2026-01-01T12:23:00Z').action, 'close');
  assert.equal(decide(data, '2026-01-01T12:23:00Z').action, 'keep'); // Production still uses days.
});
