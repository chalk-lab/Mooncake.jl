import assert from 'node:assert/strict';
import test from 'node:test';
import { configuration } from '../cli.mjs';

test('CLI defaults to dry-run and rejects invalid dates or applying without a start date', () => {
  assert.deepEqual(configuration(['--repo', 'example/repo'], {}), {
    repo: 'example/repo', since: 0, apply: false,
  });
  assert.throws(() => configuration(['--repo', 'example/repo', '--apply'], {}), /explicit --since/);
  for (const date of ['2026-02-30', '2026-1-1', 'yesterday']) {
    assert.throws(() => configuration(['--repo', 'example/repo', '--since', date], {}));
  }
  assert.equal(configuration(['--since', '2026-01-01', '--apply'], {
    GITHUB_REPOSITORY: 'example/repo',
  }).apply, true);
});
