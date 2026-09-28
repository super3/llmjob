// The marketing pages are served at extensionless URLs (/network, not
// /network.html). Builds the real site into dist/ and drives the Express app
// against it, so both halves are covered together: express.static's
// `extensions` option resolving /network to dist/network.html, and the 301 that
// retires the old .html URLs.
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

  // The chat page was removed. Its URL now serves a small page that sends the
  // visitor to the home page (site/redirects.json), so old links still work.
  it('sends the retired chat page to the home page', async () => {
    const res = await request(app).get('/chat');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<meta http-equiv="refresh" content="0; url=/" />');
    expect(res.text).toContain('location.replace("/" + location.hash)');
    expect(res.text).toContain('<meta name="robots" content="noindex" />');
    // Nothing on the site links to it any more.
    const home = await request(app).get('/');
    expect(home.text).not.toContain('href="/chat"');
  });

  // The Earn download page is the home page, and the "Run LLMs at home"
  // waitlist page that used to be the home page is /llm.
  it('serves the Earn download page as the home page', async () => {
    const res = await request(app).get('/');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<title>LLMJob Earn');
    expect(res.text).toContain('Download for Windows');
    expect(res.text).toContain('href="/llm"');
    // Both ways to get a payout address: the web wallet and the desktop wallet.
    expect(res.text).toContain('href="https://wallet.alphapool.tech/"');
    expect(res.text).toContain('href="https://github.com/pearl-research-labs/pearl/releases"');
  });

  // Download clicks are counted in Umami: every download link carries the event
  // name and which OS and which button it was, top or bottom of the page.
  it('tags every download link on the home page as an Umami event', async () => {
    const res = await request(app).get('/');
    const links = res.text.match(/<a [^>]*releases\/download\/[^>]*>/g);
    expect(links).toHaveLength(6);
    for (const a of links) {
      expect(a).toContain('data-umami-event="download"');
      expect(a).toMatch(/data-umami-event-os="(windows|linux)"/);
      expect(a).toMatch(/data-umami-event-place="(top|bottom)"/);
    }
    // Each installer's links say which OS they are.
    expect(links.filter((a) => a.includes('.exe')).every((a) => a.includes('data-umami-event-os="windows"'))).toBe(true);
    expect(links.filter((a) => a.includes('.AppImage')).every((a) => a.includes('data-umami-event-os="linux"'))).toBe(true);
  });

  // A short link for a sponsor's video: /nec lands on the home page with UTM
  // tags, so Umami can attribute the visit and its download clicks.
  it('sends the /nec short link to the home page with its UTM tags', async () => {
    const res = await request(app).get('/nec');
    expect(res.status).toBe(200);
    const target = '/?utm_source=newenglandcrypto&utm_medium=youtube&utm_campaign=pearl-miner';
    // In HTML attributes the & is escaped; in the script it is plain.
    expect(res.text).toContain('content="0; url=' + target.replace(/&/g, '&amp;') + '"');
    expect(res.text).toContain('location.replace("' + target + '" + location.hash)');
  });

  it('serves the LLM waitlist page at /llm', async () => {
    const res = await request(app).get('/llm');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<title>LLMJob — Run LLMs at home');
  });

  // Old /earn links (Discord posts, videos, the app) land on the home page, and
  // keep their #fragment, so /earn#calculator still opens the calculator.
  it('sends the old /earn URL to the home page, keeping the #fragment', async () => {
    const res = await request(app).get('/earn');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<meta http-equiv="refresh" content="0; url=/" />');
    expect(res.text).toContain('location.replace("/" + location.hash)');
    const home = await request(app).get('/');
    expect(home.text).not.toContain('href="/earn"');
  });

  it('links between pages carry no .html', async () => {
    const res = await request(app).get('/');
    expect(res.text).toContain('href="/network"');
    expect(res.text).not.toMatch(/href="[^"]*\.html"/);
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
    const res = await request(app).get('/docs.html?section=keys');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/docs?section=keys');
  });

  it('leaves non-GET requests alone', async () => {
    const res = await request(app).post('/network.html');
    expect(res.status).toBe(404);
  });
});
