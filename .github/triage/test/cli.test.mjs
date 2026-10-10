import assert from 'node:assert/strict';
import test from 'node:test';
import { configuration } from '../cli.mjs';

test('CLI defaults to dry-run and rejects invalid dates, day scaling or accidental local writes', () => {
  assert.throws(() => configuration(['--repo', 'example/repo', '--day-ms', '60000'], {}), /Unknown option/);
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
