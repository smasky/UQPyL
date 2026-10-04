// Select retained, successful CI artifacts for exactly the checked-out tag commit.
async function selectRun({github, owner, repo, sha, requestedRunId = ''}) {
  async function hasCandidate(run) {
    if (run.head_sha !== sha || run.status !== 'completed' || run.conclusion !== 'success' ||
        !['push', 'workflow_dispatch'].includes(run.event) ||
        run.path !== '.github/workflows/ci.yml' || run.repository.full_name !== `${owner}/${repo}`) {
      throw new Error('Source must be a successful CI push/manual run for the exact tag commit.');
    }
    const artifacts = await github.paginate(github.rest.actions.listWorkflowRunArtifacts,
      {owner, repo, run_id: run.id, per_page: 100});
    return artifacts.some(a => a.name === 'release-candidate' && !a.expired);
  }
  if (requestedRunId) {
    if (!/^[1-9][0-9]*$/.test(requestedRunId)) throw new Error('Invalid run ID.');
    const {data: run} = await github.rest.actions.getWorkflowRun({owner, repo, run_id: requestedRunId});
    if (!await hasCandidate(run)) throw new Error('Candidate artifact is absent or expired.');
    return run;
  }
  const runs = await github.paginate(github.rest.actions.listWorkflowRuns,
    {owner, repo, workflow_id: 'ci.yml', head_sha: sha, status: 'success', per_page: 100});
  for (const run of runs) {
    if (!['push', 'workflow_dispatch'].includes(run.event)) continue;
    if (await hasCandidate(run)) return run;
  }
  throw new Error('No tested release candidate for this SHA. Finish CI first; publishing never rebuilds.');
}
module.exports = {selectRun};
