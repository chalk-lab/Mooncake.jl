const START_MARKER = "<!-- managed-pr-summary:start -->";
const END_MARKER = "<!-- managed-pr-summary:end -->";

const sectionsConfig = [
  {
    title: "Documentation Preview",
    fallback: "_Pending. No docs preview comment found yet._",
    input: process.env.INPUT_DOCS_CONTENT,
  },
  {
    marker: "<!-- perf-results -->",
    title: "Performance",
    fallback: "_Pending. No performance comment found yet._",
    input: process.env.INPUT_PERF_CONTENT,
  },
];

function extractManagedBlock(body) {
  const start = body.indexOf(START_MARKER);
  const end = body.indexOf(END_MARKER);
  if (start === -1 || end === -1 || end < start) return "";
  return body.slice(start + START_MARKER.length, end).trim();
}

function extractManagedSections(body) {
  const managedBlock = extractManagedBlock(body);
  if (!managedBlock) return new Map();

  const sections = new Map();
  const sectionRegex = /^### (.+)$/gm;
  const matches = [...managedBlock.matchAll(sectionRegex)];
  for (let i = 0; i < matches.length; i += 1) {
    const title = matches[i][1];
    const contentStart = matches[i].index + matches[i][0].length;
    const contentEnd = i + 1 < matches.length ? matches[i + 1].index : managedBlock.length;
    const content = managedBlock.slice(contentStart, contentEnd).trim();
    if (content) sections.set(title, `### ${title}\n\n${content}`);
  }
  return sections;
}

function replaceManagedBlock(body, managedBlock) {
  const start = body.indexOf(START_MARKER);
  const end = body.indexOf(END_MARKER);

  if (start === -1 && end === -1) {
    if (!body) return managedBlock;
    if (body.endsWith("\n\n")) return `${body}${managedBlock}`;
    if (body.endsWith("\n")) return `${body}\n${managedBlock}`;
    return `${body}\n\n${managedBlock}`;
  }
  if (start !== -1 && (end === -1 || end < start)) return `${body.slice(0, start)}${managedBlock}`;
  if (start === -1 && end !== -1) return `${managedBlock}${body.slice(end + END_MARKER.length)}`;
  return `${body.slice(0, start)}${managedBlock}${body.slice(end + END_MARKER.length)}`;
}

module.exports = async ({ github, context, core }) => {
  const prNumber = context.payload.pull_request.number;
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  const pr = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: prNumber,
    per_page: 100,
  });

  const currentBody = pr.data.body || "";
  const existingSections = extractManagedSections(currentBody);
  const latestByMarker = new Map();

  for (const comment of comments) {
    const body = comment.body || "";
    const user = comment.user?.login;
    const app = comment.performed_via_github_app?.slug;
    if (user !== "github-actions[bot]" && app !== "github-actions") continue;
    for (const { marker, title } of sectionsConfig) {
      if (!marker) continue;
      if (!body.startsWith(marker)) continue;
      const content = body.slice(marker.length).trim();
      if (content) latestByMarker.set(marker, `### ${title}\n\n${content}`);
    }
  }

  const sections = sectionsConfig.map(({ marker, title, fallback, input }) => {
    if (input) return `### ${title}\n\n${input.trim()}`;
    return latestByMarker.get(marker) || existingSections.get(title) || `### ${title}\n\n${fallback}`;
  });

  const managedBlock = [
    START_MARKER,
    "## Automated PR Summary",
    "",
    ...sections,
    END_MARKER,
  ].join("\n\n");

  const updatedBody = replaceManagedBlock(currentBody, managedBlock);
  if (updatedBody === currentBody) {
    core.info("PR summary already up to date.");
    return;
  }

  await github.rest.pulls.update({ owner, repo, pull_number: prNumber, body: updatedBody });
};
