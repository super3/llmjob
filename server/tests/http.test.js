const { handle, reply, replyError } = require('../src/controllers/http');

function fakeRes() {
  return {
    statusCode: 200,
    body: undefined,
    status(code) { this.statusCode = code; return this; },
    json(body) { this.body = body; return this; },
  };
}

describe('handle', () => {
  let errSpy;
  beforeEach(() => { errSpy = jest.spyOn(console, 'error').mockImplementation(() => {}); });
  afterEach(() => { errSpy.mockRestore(); });

  it('runs the handler and leaves a normal response alone', async () => {
    const res = fakeRes();
    await handle(async (req, r) => r.json({ ok: req.x }), 'Failed')({ x: 1 }, res);
    expect(res.body).toEqual({ ok: 1 });
    expect(errSpy).not.toHaveBeenCalled();
  });

  it('passes `this` through, so it can wrap a method', async () => {
    const res = fakeRes();
    const obj = { v: 7, m: handle(async function (req, r) { r.json({ v: this.v }); }, 'Failed') };
    await obj.m({}, res);
    expect(res.body).toEqual({ v: 7 });
  });

  it('answers a deliberate 4xx with its status and message', async () => {
    const res = fakeRes();
    const err = Object.assign(new Error('Stale lock'), { status: 409 });
    await handle(async () => { throw err; }, 'Failed to do it')({}, res);
    expect(res.statusCode).toBe(409);
    expect(res.body).toEqual({ error: 'Stale lock' });
    expect(errSpy).not.toHaveBeenCalled();
  });

  // A thrown database error must not reach the caller: its message can carry
  // SQL, table names or connection details.
  it('answers anything else with a logged 500 and only the generic message', async () => {
    for (const thrown of [new Error('relation "jobs" does not exist'),
      Object.assign(new Error('upstream'), { status: 503 }),
      Object.assign(new Error('odd'), { status: 302 }),
      null]) {
      const res = fakeRes();
      await handle(async () => { throw thrown; }, 'Failed to do it')({}, res);
      expect(res.statusCode).toBe(500);
      expect(res.body).toEqual({ error: 'Failed to do it' });
    }
    expect(errSpy).toHaveBeenCalledWith('Failed to do it:', expect.anything());
  });
});

describe('reply / replyError', () => {
  it('sends a result as JSON', () => {
    const res = fakeRes();
    reply(res, { success: true });
    expect(res.statusCode).toBe(200);
    expect(res.body).toEqual({ success: true });
  });

  it("sends an error result with its own status, or 400", () => {
    const a = fakeRes();
    reply(a, { error: 'Key not found', status: 404 });
    expect(a.statusCode).toBe(404);
    expect(a.body).toEqual({ error: 'Key not found' });

    const b = fakeRes();
    replyError(b, { error: 'Bad' });
    expect(b.statusCode).toBe(400);
    expect(b.body).toEqual({ error: 'Bad' });
  });
});
