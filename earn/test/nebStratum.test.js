'use strict';

const blake3 = require('../src/shared/miner/blake3');
const s = require('../src/shared/neb/stratum');
const hdr = require('../src/shared/neb/header');
const addr = require('../src/shared/neb/address');

// A test address made with the NEB release's neb-wallet (28 Nov 2026).
const ADDRESS = 'neb1kwgy6lnm774h2cckdag40sj6sw25cpustla0gu5e37n0mn30wnzsjvwr4n';
// rabidmining's pool address, from its own API the same day.
const POOL_ADDRESS = 'neb17jws4a7qwedr42xkvk2hdq8ca6sug8v9sgt769z43h3hlnytj3wsgfhge0';

// A job as neb.rabidmining.com:3470 sent it on 28 Nov 2026 (height 13990).
const JOB = {
  job_id: 'a737f3', clean: true, version: 1,
  parent: 'ee3ffddc2705ac8f0016a990c0d5ca805282f9263fe05ddc057eafee6ada5802',
  time_start: 1795840703, time_window: 3600, bits: 3096293279,
  target: '00000000000000000000000000000000000000000000009faf8d000000000000',
  share_target: '0000000000000000000000000000000000000000000000e0f3b5110000000000',
  coinbase_prefix: '01010000000000000000000000000000000000000000000000000000000000000000ffffffff14a6360000',
  coinbase_suffix: '0210da3f710000000001f49d0af7c0765a3aa8d665957680f8eea1c41d858217ed14558de37fcc8b945d00e1f5050000000002270bbd6380c0c7022176e202083d5671ed6bb7bc0c27350bcbd134846ce87dd5',
  branch: ['5bce767d122c8d61583a9628aceae2d2fce7d42af36c625a217f523b01a0752a'],
  tx_count: 2,
  shares_root: 'd81c3a98954b6cda826d0ae634d06c9a400bef9691bd66f7fb833754429595ea',
  height: 13990, slot: 0,
  weights_hash: 'cfe112f86b5d3d825b340fe14d44435477cf4829c63603b7ac24db89d235cc5a',
  expiry: 1795841003,
};
const line = (o) => JSON.stringify(o);

describe('NEB addresses', () => {
  test('real addresses decode to a 32-byte lock and encode back', () => {
    for (const a of [ADDRESS, POOL_ADDRESS]) {
      const lock = addr.decodeNebAddress(a);
      expect(lock.length).toBe(32);
      expect(addr.encodeNebAddress(lock)).toBe(a);
      expect(addr.isNebAddress(' ' + a + ' ')).toBe(true);
    }
  });

  test('a typo, another prefix, upper case, a wrong length and no address are refused', () => {
    expect(addr.isNebAddress(ADDRESS.slice(0, -1) + 'q')).toBe(false);
    expect(addr.isNebAddress('nec1' + ADDRESS.slice(4))).toBe(false);
    expect(addr.isNebAddress(ADDRESS.toUpperCase())).toBe(false);
    expect(addr.isNebAddress(ADDRESS + 'q')).toBe(false);
    expect(addr.isNebAddress(ADDRESS.slice(0, 20) + 'b' + ADDRESS.slice(21))).toBe(false);
    expect(addr.isNebAddress(null)).toBe(false);
  });

  test('padding bits that are not zero are refused, even under a good checksum', () => {
    const groups = [...ADDRESS.slice(4, -6)].map((c) => addr.CHARSET.indexOf(c));
    expect(addr.encodeGroups(groups)).toBe(ADDRESS);
    const padded = groups.slice(0, -1).concat([groups[51] | 1]);
    const a = addr.encodeGroups(padded);
    expect(a.length).toBe(addr.ADDRESS_LENGTH);
    expect(addr.isNebAddress(a)).toBe(false);
  });

  test('shortens for the log', () => {
    expect(addr.shortenNebAddress(ADDRESS)).toBe('neb1kwgy…vwr4n');
    expect(addr.shortenNebAddress('short')).toBe('short');
    expect(addr.shortenNebAddress(null)).toBe('');
  });
});

describe('requests', () => {
  test('subscribe names the agent and the protocol', () => {
    expect(s.buildSubscribe('llmjob-earn/1.2.3')).toEqual({ id: 1, method: 'mining.subscribe', params: ['llmjob-earn/1.2.3', 'neb-pool/1'] });
    expect(s.buildSubscribe().params[0]).toBe('llmjob-earn');
  });

  test('the login is address.worker, the worker cut to what the pool takes', () => {
    expect(s.loginOf(ADDRESS, 'rig 01!')).toBe(ADDRESS + '.rig-01-');
    expect(s.loginOf(' ' + ADDRESS, 'x'.repeat(40))).toBe(ADDRESS + '.' + 'x'.repeat(32));
    expect(s.loginOf(ADDRESS, null)).toBe(ADDRESS);
    expect(s.loginOf(null, '')).toBe('');
    expect(s.buildAuthorize('a.b')).toEqual({ id: 2, method: 'mining.authorize', params: ['a.b', 'x'] });
  });

  test('a submit is login, job, extranonce2, time, nonce', () => {
    expect(s.buildSubmit(10, 'a.b', { jobId: 'a737f3', extranonce2: '00'.repeat(12), time: 5, nonce: 30817 }))
      .toEqual({ id: 10, method: 'mining.submit', params: ['a.b', 'a737f3', '00'.repeat(12), 5, 30817] });
    expect(s.encode({ a: 1 })).toBe('{"a":1}\n');
  });
});

describe('parseMessage', () => {
  test('the subscribe answer', () => {
    const m = s.parseMessage(line({
      id: 1, error: null,
      result: { session: 'f2f3b4ac', extranonce1: 'ff681386', extranonce2_size: 12, chain_id: 'DA2F', dataset_hash: 'AB'.repeat(32) },
    }));
    expect(m).toEqual({
      kind: 'subscribed', session: 'f2f3b4ac', extranonce1: Buffer.from('ff681386', 'hex'), extranonce2Size: 12,
      chainId: 'da2f', datasetHash: 'ab'.repeat(32),
    });
    expect(s.parseMessage(line({ id: 1, result: { extranonce1: 'ff', extranonce2_size: 4, dataset_hash: 'ab'.repeat(32) } })).session).toBe('');
  });

  test('a subscribe answer without what the miner needs', () => {
    const fail = (result, error) => s.parseMessage(line({ id: 1, result, error }));
    expect(fail(null, [20, 'no']).error).toEqual({ code: 20, message: 'no' });
    for (const r of [null, 5, { extranonce1: 'zz', extranonce2_size: 12, dataset_hash: 'ab'.repeat(32) },
      { extranonce1: '', extranonce2_size: 12, dataset_hash: 'ab'.repeat(32) },
      { extranonce1: 'ff', extranonce2_size: 0, dataset_hash: 'ab'.repeat(32) },
      { extranonce1: 'ff', extranonce2_size: 12, dataset_hash: 'ab' }]) {
      expect(fail(r, null)).toEqual({ kind: 'subscribe-fail', error: { code: null, message: 'the pool\'s answer names no extranonce or dataset' } });
    }
  });

  test('the authorize answer', () => {
    expect(s.parseMessage(line({ id: 2, result: true, error: null }))).toEqual({ kind: 'authorized' });
    expect(s.parseMessage(line({ id: 2, result: null, error: [24, 'bad address'] }))).toEqual({ kind: 'login-fail', error: { code: 24, message: 'bad address' } });
    expect(s.parseMessage(line({ id: 2, result: false, error: null })).error.message).toBe('the pool did not authorize the login');
  });

  test('the target and the model', () => {
    expect(s.parseMessage(line({ id: null, method: 'mining.set_target', params: ['FF'.repeat(32)] })))
      .toEqual({ kind: 'target', target: Buffer.alloc(32, 0xff) });
    expect(s.parseMessage(line({ id: null, method: 'mining.set_target', params: ['ff'] })).kind).toBe('unknown');
    expect(s.parseMessage(line({ id: null, method: 'mining.set_model', params: [1, 'ab'.repeat(32), '0f10'] })))
      .toEqual({ kind: 'model', slot: 1, weightsHash: 'ab'.repeat(32), weights: Buffer.from('0f10', 'hex') });
    for (const params of [['1', 'ab'.repeat(32), '0f'], [1, 'ab', '0f'], [1, 'ab'.repeat(32), 'xz'], [1, 'ab'.repeat(32), ''], undefined]) {
      expect(s.parseMessage(line({ id: null, method: 'mining.set_model', params })).kind).toBe('bad-model');
    }
  });

  test('a job', () => {
    const m = s.parseMessage(line({ id: null, method: 'mining.notify', params: [JOB] }));
    expect(m.kind).toBe('job');
    expect(m).toMatchObject({
      jobId: 'a737f3', clean: true, version: 1, timeStart: 1795840703, timeWindow: 3600, bits: 3096293279, txCount: 2,
      height: 13990, slot: 0, weightsHash: JOB.weights_hash, expiry: 1795841003,
    });
    expect(m.parent.toString('hex')).toBe(JOB.parent);
    expect(m.blockTarget.toString('hex')).toBe(JOB.target);
    expect(m.shareTarget.toString('hex')).toBe(JOB.share_target);
    expect(m.branch.map((b) => b.toString('hex'))).toEqual(JOB.branch);
    const bare = s.parseJob(Object.assign({}, JOB, { clean: 1, share_target: undefined, height: undefined, expiry: undefined }));
    expect(bare).toMatchObject({ kind: 'job', clean: false, shareTarget: null, height: null, expiry: null });
  });

  test('a job this miner cannot build a header from', () => {
    const bad = (change) => s.parseJob(Object.assign({}, JOB, change));
    expect(s.parseJob(null)).toEqual({ kind: 'bad-job', jobId: null, reason: 'the job is not an object' });
    expect(s.parseJob([JOB]).reason).toBe('the job is not an object');
    expect(bad({ job_id: undefined })).toEqual({ kind: 'bad-job', jobId: null, reason: 'no job_id' });
    expect(bad({ bits: -1 }).reason).toBe('bits is not a 32-bit number');
    expect(bad({ time_start: 2 ** 32 }).reason).toBe('time_start is not a 32-bit number');
    expect(bad({ version: '1' }).reason).toBe('version is not a 32-bit number');
    expect(bad({ parent: 'ee' })).toEqual({ kind: 'bad-job', jobId: 'a737f3', reason: 'parent is not 32 bytes of hex' });
    expect(bad({ weights_hash: 5 }).reason).toBe('weights_hash is not 32 bytes of hex');
    expect(bad({ coinbase_prefix: 'xyz' }).reason).toBe('coinbase_prefix is not hex');
    expect(bad({ branch: 'no' }).reason).toBe('branch is not a list of 32-byte values');
    expect(bad({ branch: ['ab'] }).reason).toBe('branch is not a list of 32-byte values');
    expect(bad({ slot: 300 }).reason).toBe('slot is not a dataset index');
    expect(bad({ slot: 0.5 }).reason).toBe('slot is not a dataset index');
  });

  test('submit answers', () => {
    expect(s.parseMessage(line({ id: 10, result: true, error: null }))).toEqual({ kind: 'submit-accepted', id: 10 });
    expect(s.parseMessage(line({ id: 11, result: null, error: [23, 'ticket above the pool target'] })))
      .toEqual({ kind: 'submit-rejected', id: 11, error: { code: 23, message: 'ticket above the pool target' } });
    expect(s.parseMessage(line({ id: 12, result: false })).error.message).toBe('no reason given (result false)');
    expect(s.errorText({ code: 22, message: 'duplicate' })).toBe('[22] duplicate');
  });

  test('what is not a message this miner knows', () => {
    expect(s.parseMessage('')).toEqual({ kind: 'empty' });
    expect(s.parseMessage(null)).toEqual({ kind: 'empty' });
    expect(s.parseMessage('{nope').kind).toBe('unparseable');
    expect(s.parseMessage('x'.repeat(s.MAX_LINE + 1))).toEqual({ kind: 'unparseable', raw: 'x'.repeat(200) });
    expect(s.parseMessage('[1]').kind).toBe('unknown');
    expect(s.parseMessage('5').kind).toBe('unknown');
    expect(s.parseMessage(line({ id: null, method: 'mining.ping', params: [] })).kind).toBe('unknown');
    expect(s.parseMessage(line({ id: 9 })).kind).toBe('unknown');
    expect(s.parseMessage(line({ id: null, result: true })).kind).toBe('unknown');
    expect(s.parseMessage(line({ id: null, result: null, error: [20, 'other'] }))).toEqual({ kind: 'pool-error', error: { code: 20, message: 'other' } });
  });
});

describe('the header of an attempt', () => {
  const job = s.parseJob(JOB);
  const en1 = Buffer.from('ff681386', 'hex');
  const en2 = Buffer.alloc(12, 1);

  test('follows the pool protocol\'s recipe', () => {
    const coinbase = Buffer.concat([job.coinbasePrefix, en1, en2, job.coinbaseSuffix]);
    const H = (tag, ...parts) => blake3.hash(Buffer.concat([Buffer.from(tag), ...parts]));
    let h = H('leaf', blake3.hash(coinbase));
    h = H('node', h, job.branch[0]);
    const le = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); return b; };
    const root = H('body', le(2), h, job.sharesRoot);
    expect(hdr.bodyRoot(job, en1, en2).equals(root)).toBe(true);
    const header = hdr.buildHeader(job, en1, en2, job.timeStart);
    expect(header.length).toBe(hdr.HEADER_BYTES);
    expect(header.equals(Buffer.concat([le(1), job.parent, root, le(job.timeStart), le(job.bits), le(0)]))).toBe(true);
  });

  test('the nonce is the last word, and the key is the header\'s BLAKE3', () => {
    const header = hdr.buildHeader(job, en1, en2, job.timeStart);
    const h = hdr.withNonce(header, 0xdeadbeef);
    expect(h.readUInt32LE(hdr.NONCE_OFFSET)).toBe(0xdeadbeef);
    expect(header.readUInt32LE(hdr.NONCE_OFFSET)).toBe(0);
    expect(hdr.attemptKey(header, 7).equals(blake3.hash(hdr.withNonce(header, 7)))).toBe(true);
  });
});
