// The marketing pages are served at extensionless URLs (/network, not
// /network.html). Builds the real site into dist/ and drives the Express app
// against it, so both halves are covered together: express.static's `extensions`
// option resolving /network to dist/network.html, and the 301 that retires the
// old .html URLs.
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const request = require('supertest');
const { app } = require('../src/index');

const ROOT = path.join(__dirname, '../..');

beforeAll(() => {
  execFileSync('node', ['site/build-site.mjs'], { cwd: ROOT, stdio: 'ignore' });
});

describe('extensionless page URLs', () => {
  it('serves a page at its extensionless path', async () => {
    const res = await request(app).get('/network');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/text\/html/);
    expect(res.text).toContain('<title>LLMJob Network');
  });

  it('serves the download page as the home page', async () => {
    const res = await request(app).get('/');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<title>LLMJob — Free Pearl (PRL) GUI Miner');
    expect(res.text).toContain('Earnings calculator');
  });

  it('serves the managed-mining waitlist page', async () => {
    const res = await request(app).get('/managed');
    expect(res.status).toBe(200);
    expect(res.text).toContain('id="wl-form"');
    expect(res.text).toContain("'/api/waitlist'");
  });

  it('links between pages carry no .html', async () => {
    const res = await request(app).get('/');
    expect(res.text).toContain('href="/managed"');
    expect(res.text).toContain('href="/network"');
    expect(res.text).not.toMatch(/href="[^"]*\.html"/);
  });
});

// Pages the pivot retired still answer, with a stub that sends the visitor to
// the page that replaced them — /earn is linked from videos, /chat from Discord.
describe('retired pages', () => {
  it.each(['/earn', '/chat', '/docs', '/dashboard', '/add-node'])('%s is a redirect stub to the home page', async (url) => {
    const res = await request(app).get(url);
    expect(res.status).toBe(200);
    expect(res.text).toContain('<meta http-equiv="refresh" content="0; url=/" />');
    expect(res.text).toContain('location.replace("/" + location.hash)');
    expect(res.text).toContain('noindex');
  });
});

describe('legacy .html URLs', () => {
  it('redirects /network.html to /network', async () => {
    const res = await request(app).get('/network.html');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/network');
  });

  it('redirects /index.html to the root', async () => {
    const res = await request(app).get('/index.html');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/');
  });

  it('keeps the query string', async () => {
    const res = await request(app).get('/network.html?address=prl1pabc');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/network?address=prl1pabc');
  });

  it('leaves non-GET requests alone', async () => {
    const res = await request(app).post('/network.html');
    expect(res.status).toBe(404);
  });
});
