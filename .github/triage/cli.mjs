import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { githubApi, runTriage } from './github.mjs';
import { parseApproved, timestamp } from './policy.mjs';

export function configuration(args, env) {
  const { values } = parseArgs({ args, options: {
    repo: { type: 'string', default: env.GITHUB_REPOSITORY },
    since: { type: 'string' }, apply: { type: 'boolean', default: false },
  } });
  if (!/^[\w.-]+\/[\w.-]+$/.test(values.repo || '')) throw new Error('Supply --repo OWNER/REPO');
  if (values.apply && !values.since) {
    throw new Error('--apply requires an explicit --since date');
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
    const approved = parseApproved(readFileSync(new URL('contributors.txt', import.meta.url), 'utf8'));
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
