const ContainerService = require('../services/containerService');
const { handle, reply } = require('./http');

// Rented containers on our own rigs (issue #318). The admin calls mirror
// Salad's container-group calls: create, status, logs, delete. The rig calls
// are signed by the rig's node key (verifySignature), and the node id it names
// must be the one that key mints, so no one can take or end another rig's runs.

const service = (req) => new ContainerService(req.app.locals.db);
const actor = (req) => req.user.id;

// ── Admin ────────────────────────────────────────────────────────────────────

const createJob = handle(async (req, res) => {
  const b = req.body || {};
  const out = await service(req).createJob({
    name: b.name, image: b.image, env: b.env, target: b.target,
    durationSec: b.durationSec, replicas: b.replicas, payRateUsdHour: b.payRateUsdHour,
  }, actor(req));
  if (out.error) return reply(res, out);
  res.status(201).json(out);
}, 'Failed to create container job');

const listJobs = handle(async (req, res) => {
  reply(res, await service(req).listJobs(req.query.limit));
}, 'Failed to list container jobs');

const getJob = handle(async (req, res) => {
  reply(res, await service(req).getJob(req.params.id));
}, 'Failed to get container job');

const getLogs = handle(async (req, res) => {
  reply(res, await service(req).getLogs(req.params.id, {
    since: req.query.since, limit: req.query.limit, runId: req.query.runId,
  }));
}, 'Failed to get container logs');

const deleteJob = handle(async (req, res) => {
  reply(res, await service(req).deleteJob(req.params.id));
}, 'Failed to delete container job');

const ledger = handle(async (req, res) => {
  reply(res, await service(req).ledger({ nodeId: req.query.nodeId }));
}, 'Failed to read the minutes ledger');

const listRigs = handle(async (req, res) => {
  reply(res, await service(req).listRigs());
}, 'Failed to list rigs');

// ── Rig ──────────────────────────────────────────────────────────────────────

// The node id the signature was made for, when it belongs to the signing key.
function rigId(req, res) {
  const v = req.verifiedNode;
  if (!ContainerService.nodeIdMatchesKey(v.nodeId, v.publicKey)) {
    res.status(401).json({ error: 'Node id does not match its key' });
    return null;
  }
  return v.nodeId;
}

const rigPoll = handle(async (req, res) => {
  const nodeId = rigId(req, res);
  if (!nodeId) return;
  reply(res, await service(req).poll(nodeId, req.body));
}, 'Failed to poll for a container job');

const rigHeartbeat = handle(async (req, res) => {
  const nodeId = rigId(req, res);
  if (!nodeId) return;
  reply(res, await service(req).heartbeat(nodeId, req.params.runId, req.body));
}, 'Failed to record a heartbeat');

const rigEnd = handle(async (req, res) => {
  const nodeId = rigId(req, res);
  if (!nodeId) return;
  reply(res, await service(req).endRun(nodeId, req.params.runId, req.body));
}, 'Failed to end the run');

module.exports = {
  createJob, listJobs, getJob, getLogs, deleteJob, ledger, listRigs, rigPoll, rigHeartbeat, rigEnd,
};
