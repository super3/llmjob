// The marketing pages are served at extensionless URLs (/network, not
// /network.html). Builds the real site into dist/ and drives the Express app
// against it, so both halves are covered together: express.static's
// `extensions` option resolving /network to dist/network.html, and the 301 that
// retires the old .html URLs.
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const request = require('supertest');
const { app } = require('../src/index');

const ROOT = path.join(__dirname, '../..');

beforeAll(() => {
  execFileSync('node', ['site/build-site.mjs'], { cwd: ROOT, stdio: 'ignore' });
});

// Run a redirect page's script against a pretend browser location and return
// where it sends the visitor, so the tests check the behaviour, not the text.
function redirectFrom(html, search = '', hash = '') {
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  let to = null;
  vm.runInNewContext(script, { location: { search, hash, replace: (url) => { to = url; } } });
  return to;
}

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
    expect(redirectFrom(res.text)).toBe('/');
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
    expect(redirectFrom(res.text)).toBe(target);
    // Anything the visitor's link carried is added after the short link's own tags.
    expect(redirectFrom(res.text, '?fbclid=abc', '#calculator')).toBe(target + '&fbclid=abc#calculator');
  });

  it('sends the /rabid short link to the home page with its UTM tags', async () => {
    const res = await request(app).get('/rabid');
    expect(res.status).toBe(200);
    expect(redirectFrom(res.text)).toBe('/?utm_source=rabid&utm_medium=youtube&utm_campaign=pearl-miner');
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
    expect(redirectFrom(res.text)).toBe('/');
    expect(redirectFrom(res.text, '', '#calculator')).toBe('/#calculator');
    const home = await request(app).get('/');
    expect(home.text).not.toContain('href="/earn"');
  });

  // A sponsor's July video still links to /earn.html?ref=rabid. The tag has to
  // survive both hops (the .html 301, then the redirect page) to reach Umami.
  it('keeps the ?query of a tagged link through the /earn redirect', async () => {
    const hop = await request(app).get('/earn.html?ref=rabid');
    expect(hop.status).toBe(301);
    expect(hop.headers.location).toBe('/earn?ref=rabid');
    const res = await request(app).get('/earn');
    expect(redirectFrom(res.text, '?ref=rabid')).toBe('/?ref=rabid');
    expect(redirectFrom(res.text, '?ref=rabid', '#calculator')).toBe('/?ref=rabid#calculator');
  });

  it('links between pages carry no .html', async () => {
    const res = await request(app).get('/');
    expect(res.text).toContain('href="/network"');
    expect(res.text).not.toMatch(/href="[^"]*\.html"/);
  });
});

// The home page carries the HiveOS flight sheet. The Installation URL must name
// the tarball CI publishes: earn/scripts/build-hiveos.mjs names it after
// earn/package.json's version, and the page stamps appVersion from
// site/config.json. If the two ever differ, the page points at a file that the
// release doesn't have.
describe('HiveOS flight sheet on the home page', () => {
  const { version } = JSON.parse(fs.readFileSync(path.join(ROOT, 'earn/package.json'), 'utf8'));
  const manifest = fs.readFileSync(path.join(ROOT, 'earn/hiveos/h-manifest.conf'), 'utf8');
  const minerName = manifest.match(/^CUSTOM_NAME=(.+)$/m)[1];

  async function section() {
    const res = await request(app).get('/');
    const m = res.text.match(/<section[^>]*id="hiveos"[^>]*>([\s\S]*?)<\/section>/);
    expect(m).not.toBeNull();
    return { page: res.text, sec: m[1] };
  }

  it('gives the versioned package URL of the current release', async () => {
    const { sec } = await section();
    const url = `https://github.com/super3/llmjob/releases/download/v${version}/${minerName}-${version}.tar.gz`;
    expect(sec).toContain(`<code id="hive-url">${url}</code>`);
    // Not the unversioned name: HiveOS would keep the old build on update.
    expect(sec).not.toContain('llmjob-earn-hiveos.tar.gz');
    expect(sec).not.toContain('{{');
  });

  it('lists the flight sheet fields', async () => {
    const { sec } = await section();
    for (const field of ['Miner', 'Miner name', 'Installation URL', 'Hash algorithm',
      'Wallet and worker template', 'Pool URL', 'Pass', 'Extra config arguments']) {
      expect(sec).toContain(`<span class="hive-k">${field}</span>`);
    }
    expect(sec).toContain('<code>Custom</code>');
    expect(sec).toContain(`<code>${minerName}</code>`);
    expect(sec).toContain('<code>pearlhash</code>');
    expect(sec).toContain('<code>%WAL%</code>');
    expect(sec).toContain('<code>%WAL%.%WORKER_NAME%</code>');
    expect(sec).toContain('<code>us.pearl.herominers.com:1200</code>');
    expect(sec).toContain('Ubuntu 22.04');
  });

  it('is linked from both download menus', async () => {
    const { page } = await section();
    expect(page.match(/<a role="menuitem" href="#hiveos">/g)).toHaveLength(2);
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
