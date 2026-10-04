const {test} = require('node:test');
const assert = require('node:assert/strict');
const {selectRun} = require('../scripts/select_release_run.cjs');
const base = {id: 42, head_sha: 'abc', status: 'completed', conclusion: 'success',
  event: 'push', path: '.github/workflows/ci.yml', repository: {full_name: 'owner/repo'}};
function client(runs, artifacts = [{name: 'release-candidate', expired: false}]) {
  const rest = {actions: {listWorkflowRuns: 'runs', listWorkflowRunArtifacts: 'artifacts',
    getWorkflowRun: async () => ({data: runs[0]})}};
  return {rest, paginate: async (method, args) => {
    if (method === 'runs') {
      assert.equal(args.head_sha, 'abc');
      assert.equal(args.workflow_id, 'ci.yml');
      return runs;
    }
    return typeof artifacts === 'function' ? artifacts(args.run_id) : artifacts;
  }};
}
function select(github, requestedRunId = '') {
  return selectRun({github, owner: 'owner', repo: 'repo', sha: 'abc', requestedRunId});
}
test('select a successful matching candidate', async () => {
  assert.equal((await select(client([base]))).id, 42);
});
test('explicit run ID works', async () => {
  assert.equal((await select(client([base]), '42')).id, 42);
});
for (const [key, value] of Object.entries({head_sha: 'other', status: 'in_progress', conclusion: 'failure',
    event: 'pull_request', path: '.github/workflows/other.yml', repository: {full_name: 'fork/repo'}})) {
  test(`reject explicit source with wrong ${key}`, async () => {
    await assert.rejects(select(client([{...base, [key]: value}]), '42'), /successful CI/);
  });
}
test('reject malformed run ID', async () => {
  await assert.rejects(select(client([base]), '42;bad'), /Invalid run ID/);
});
test('reject expired explicit artifact', async () => {
  await assert.rejects(select(client([base], [{name: 'release-candidate', expired: true}]), '42'), /expired/);
});
test('skip expired latest run and reuse older same-SHA candidate', async () => {
  const github = client([{...base, id: 43}, base], id => id === 43 ? [] : [{name: 'release-candidate', expired: false}]);
  assert.equal((await select(github)).id, 42);
});
test('PR artifacts are never automatically released', async () => {
  await assert.rejects(select(client([{...base, event: 'pull_request'}])), /No tested release candidate/);
});
test('missing candidate fails without starting a build', async () => {
  await assert.rejects(select(client([])), /publishing never rebuilds/);
});
