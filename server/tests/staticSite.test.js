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

  // The font URLs live in site/config.json. Pages that never loaded IBM Plex
  // Mono's 600 weight still don't, and every template token is rendered.
  it('loads each page\'s fonts from the shared config, with no template left over', async () => {
    const config = JSON.parse(fs.readFileSync(path.join(ROOT, 'site', 'config.json'), 'utf8'));
    const home = await request(app).get('/');
    expect(home.text).toContain(`<link href="${config.fonts}" rel="stylesheet" />`);
    const network = await request(app).get('/network');
    expect(network.text).toContain(`<link href="${config.fontsNoMono600}" rel="stylesheet" />`);
    expect(config.fontsNoMono600).toContain('IBM+Plex+Mono:wght@400;500&');
    for (const page of ['/', '/network']) {
      const res = await request(app).get(page);
      expect(res.text).not.toMatch(/\{\{[>#^!/]/);
    }
  });

  // The add-node page only pointed at the dashboard's join token, so its old
  // links now go straight there.
  it('sends the retired add-node page to the dashboard', async () => {
    const res = await request(app).get('/add-node');
    expect(res.status).toBe(200);
    expect(redirectFrom(res.text)).toBe('/dashboard');
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

  // Download clicks are counted in Umami, with which OS and which button it
  // was, top or bottom of the page. The two Download buttons download; the
  // Windows and Linux items in their menus only pick the platform the button
  // gets, so they count as a pick, not as a download.
  it('tags every download link on the home page as an Umami event', async () => {
    const res = await request(app).get('/');
    const links = res.text.match(/<a [^>]*releases\/download\/[^>]*>/g);
    expect(links).toHaveLength(6);
    for (const a of links) {
      expect(a).toMatch(/data-umami-event-os="(windows|linux)"/);
      expect(a).toMatch(/data-umami-event-place="(top|bottom)"/);
    }
    const buttons = links.filter((a) => a.includes('data-dl-main'));
    expect(buttons).toHaveLength(2);
    for (const a of buttons) expect(a).toContain('data-umami-event="download"');
    for (const a of links.filter((x) => !buttons.includes(x))) expect(a).toContain('data-umami-event="download-platform"');
    // Each installer's links say which OS they are.
    expect(links.filter((a) => a.includes('.exe')).every((a) => a.includes('data-umami-event-os="windows"'))).toBe(true);
    expect(links.filter((a) => a.includes('.AppImage')).every((a) => a.includes('data-umami-event-os="linux"'))).toBe(true);
  });

  // The calculator's popular-card chips look their card up by name in the
  // benchmark list, so a chip whose name isn't in it would break the section.
  it('has a benchmark for every popular card in the calculator', async () => {
    const res = await request(app).get('/');
    const list = vm.runInNewContext(res.text.match(/const CARDS = (\[[\s\S]*?\])\.map/)[1]);
    const popular = vm.runInNewContext(res.text.match(/const POPULAR = (\[[^\]]*\])/)[1]);
    expect(list.length).toBeGreaterThan(60);
    for (const [name, th, arch] of list) {
      expect(typeof name).toBe('string');
      expect(th).toBeGreaterThan(0);
      expect(arch).toMatch(/^(NVIDIA|AMD) · /);
    }
    for (const name of popular) expect(list.map((c) => c[0])).toContain(name);
  });

  // The page's HTML carries the figures a visitor sees before any script runs
  // (and the ones a crawler or a link preview keeps): the calculator's result
  // for one RTX 4090, the price it is based on, and the app preview's numbers.
  // The script draws the same figures from its snapshot (`est`). When the
  // snapshot is refreshed the HTML has to follow: on 11 November it was left
  // at the old price, $213.72 a month beside "Per day $5.26".
  it('writes the calculator\'s first figures from the same snapshot its script uses', async () => {
    const res = await request(app).get('/');
    const calc = vm.runInNewContext(
      res.text.match(/const est = [\s\S]*?const money = [^\n]*\n/)[0] + '({ est, prlPerDay, usdPerDay, fmt, money })');
    const list = vm.runInNewContext(res.text.match(/const CARDS = (\[[\s\S]*?\])\.map/)[1]);
    // The card the calculator opens on.
    const name = res.text.match(/pick\.card = byName\('([^']+)'\)/)[1];
    const th = list.find((c) => c[0] === name)[1];
    const shown = (id) => res.text.match(new RegExp('id="' + id + '"[^>]*>([^<]*)<'))[1];
    const day = calc.usdPerDay(th);
    expect(shown('out-label')).toBe(name + ' · ' + calc.fmt(th, 1) + ' TH/s');
    expect(shown('out-month')).toBe(calc.money(day * 30));
    expect(shown('out-day')).toBe(calc.money(day));
    expect(shown('out-prl')).toBe(calc.fmt(calc.prlPerDay(th) * 30, 1));
    expect(shown('calc-price')).toBe('$' + calc.est.PRICE.toFixed(2));
    expect(shown('pv-th')).toBe(calc.fmt(th, 1));
    expect(shown('pv-day')).toBe(calc.money(day));
    // The preview's balance is a fixed 128.41 PRL at the snapshot's price.
    expect(res.text).toContain('<b>128.41 PRL</b>');
    expect(shown('pv-bal')).toBe(' ≈ ' + calc.money(128.41 * calc.est.PRICE));
    // The note shown when live data does not load names the same price.
    expect(res.text).toMatch(new RegExp('uses a snapshot of [^(]*\\(\\$' + calc.est.PRICE.toFixed(2).replace('.', '\\.') + '/PRL\\)'));
  });

  // "N GPUs mining now" is the board's exact count of online GPUs. Without an
  // answer it shows no number at all, never a made-up one.
  it('shows the exact number of GPUs mining from the network board', async () => {
    const res = await request(app).get('/');
    expect(res.text).toContain('<b id="gpu-count">GPUs</b> mining now');
    const script = res.text.match(/\(function \(\) \{\s*const el = document\.getElementById\('gpu-count'\);[\s\S]*?\}\)\(\);/)[0];
    async function shown(reply) {
      const el = { textContent: 'GPUs' };
      const fetch = async () => (reply ? { ok: true, json: async () => reply } : Promise.reject(new Error('offline')));
      vm.runInNewContext(script, { document: { getElementById: () => el }, fetch, API_BASE: '' });
      await new Promise((r) => setImmediate(r));
      return el.textContent;
    }
    expect(await shown({ totalWorkers: 64 })).toBe('64 GPUs');
    expect(await shown({ totalWorkers: 1 })).toBe('1 GPU');
    expect(await shown({ totalWorkers: 0 })).toBe('GPUs');
    expect(await shown(null)).toBe('GPUs');
  });

  // Every public page carries the same header and footer.
  it('gives every public page the shared header and footer', async () => {
    for (const page of ['/', '/llm', '/network', '/docs', '/terms', '/privacy', '/hiveos-llm', '/benchmarks']) {
      const res = await request(app).get(page);
      expect(res.text).toContain('<nav class="nav">');
      expect(res.text).toContain('<footer class="foot">');
      for (const href of ['/network', '/llm', '/benchmarks', '/terms', '/privacy', 'https://github.com/super3/llmjob']) {
        expect(res.text).toContain(`href="${href}"`);
      }
      // The header's link for the page you're on is the highlighted one.
      const active = res.text.match(/class="nav-link active" href="([^"]*)"/);
      expect(active ? active[1] : null).toBe(['/', '/network'].includes(page) ? page : null);
    }
  });

  // The HiveOS LLM guide is linked from the home page's HiveOS section, not the footer.
  it('links the HiveOS LLM guide from the home page only', async () => {
    expect((await request(app).get('/')).text).toContain('href="/hiveos-llm"');
    expect((await request(app).get('/network')).text).not.toContain('href="/hiveos-llm"');
  });

  // Issue #308: Hive Shell's web terminal broke the guide's curl, written over
  // three lines joined with "\", and hung. Every command there is one line now,
  // and its Copy button copies exactly that line.
  it('gives the HiveOS LLM guide one-line commands with a Copy button each', async () => {
    const text = (await request(app).get('/hiveos-llm')).text;
    const cmds = [...text.matchAll(/<div class="cmd"><pre id="([^"]+)">([^<]*)<\/pre><button class="cmd-copy" type="button" data-copy="([^"]+)">Copy<\/button><\/div>/g)];
    expect(cmds.map((c) => c[1])).toEqual(['cmd-rig', 'cmd-lan']);
    for (const [, id, cmd, target] of cmds) {
      expect(target).toBe(id);
      expect(cmd.startsWith('curl ')).toBe(true);
      expect(cmd).not.toMatch(/[\n\\]/);
    }
    // Every curl on the page is one of those.
    expect(text.match(/curl /g)).toHaveLength(2);
    expect(cmds[0][2]).toContain('http://127.0.0.1:8000/v1/chat/completions');
    expect(cmds[1][2]).toContain(':8000/v1/models');
    // The button copies the command's text, trimmed, so no newline comes with it.
    expect(text).toContain('navigator.clipboard.writeText(el.textContent.trim())');
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

  // A second video from the same channel: same source, its own campaign, so
  // Umami shows the two videos apart.
  it('sends the /rabid2 short link to the home page with its own campaign', async () => {
    const res = await request(app).get('/rabid2');
    expect(res.status).toBe(200);
    expect(redirectFrom(res.text)).toBe('/?utm_source=rabid&utm_medium=youtube&utm_campaign=pearl-miner-2');
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

// The home page carries the HiveOS flight sheet. Its Installation URL is the
// latest release's unversioned package, the same one h-manifest.conf names: a
// flight sheet installs it once, and the miner updates itself after that
// (earn/src/cli/hiveos.js).
describe('HiveOS flight sheet on the home page', () => {
  const manifest = fs.readFileSync(path.join(ROOT, 'earn/hiveos/h-manifest.conf'), 'utf8');
  const minerName = manifest.match(/^CUSTOM_NAME=(.+)$/m)[1];
  const URL = `https://github.com/super3/llmjob/releases/latest/download/${minerName}-hiveos.tar.gz`;

  // The section's HTML with the <wbr> line-break hints taken out: they add no
  // text, so a copied or selected URL doesn't carry them.
  async function section() {
    const res = await request(app).get('/');
    const m = res.text.match(/<section[^>]*id="hiveos"[^>]*>([\s\S]*?)<\/section>/);
    expect(m).not.toBeNull();
    return { page: res.text, sec: m[1].replace(/<wbr>/g, '') };
  }

  // The text of the <code> with this id, which is what its Copy button copies.
  function codeText(sec, id) {
    const m = sec.match(new RegExp(`<code id="${id}"[^>]*>([\\s\\S]*?)</code>`));
    expect(m).not.toBeNull();
    return m[1].replace(/<[^>]+>/g, '');
  }

  it('gives the package URL that never changes, the same as the manifest', async () => {
    const { sec } = await section();
    expect(codeText(sec, 'hive-url')).toBe(URL);
    expect(manifest).toContain(URL);
    // Not a versioned URL: the page would have to change on every release.
    expect(sec).not.toMatch(/releases\/download\/v/);
    expect(sec).not.toContain('{{');
  });

  // The copy script finds each field by its button's data-copy id, and does
  // nothing for an id no element has.
  it('has a Copy button on each field to fill in', async () => {
    const { sec } = await section();
    const buttons = sec.match(/<button [^>]*data-copy="[^"]*"[^>]*>/g);
    const ids = buttons.map((b) => b.match(/data-copy="([^"]*)"/)[1]);
    expect(ids).toEqual(['hive-name', 'hive-url', 'hive-wallet', 'hive-pool']);
    expect(codeText(sec, 'hive-name')).toBe(minerName);
    expect(codeText(sec, 'hive-pool')).toBe('us.pearl.herominers.com:1200');
    // The wallet template is filled in from the address box.
    expect(sec).toContain('<input class="hive-addr" id="hive-addr"');
    expect(buttons[2]).toContain('data-needs="hive-addr"');
  });

  it('lists the four fields and says to leave the rest blank', async () => {
    const { sec } = await section();
    for (const field of ['Miner', 'Miner name', 'Installation URL', 'Wallet and worker template', 'Pool URL', 'Everything else']) {
      expect(sec).toContain(`<span class="hive-k">${field}</span>`);
    }
    expect(sec).toContain('<code>Custom</code>');
    expect(sec).toContain('<code>pearlhash</code>');
    expect(sec).toContain('<code>--no-update</code>');
    // Pool URL is read: a HeroMiners address or another pool's host:port.
    expect(sec).toContain('To mine on another pool, put its <code>host:port</code> here.');
    expect(sec).toContain('<code>prl-us.kryptex.network:7048</code>');
    expect(sec).toContain('Ubuntu 20.04');
  });

  // What the hint under the button says for what was pasted, from the page's own
  // function. Issue #308: a Kryptex ID got the generic "doesn't look like a
  // Pearl address", which read as the page doing nothing. Kryptex accounts are
  // going away, so the page says that and points to the prl1p… address.
  it('says why it makes no flight sheet, and sends Kryptex accounts to a prl1p… address', async () => {
    const { page } = await section();
    const src = page.match(/function addressProblem\(v\) \{[\s\S]*?\n {4}\}/)[0];
    const why = (v) => vm.runInNewContext(src + '; addressProblem(v)', { v });
    expect(why('prl1pexample0123456789abcdef')).toBeNull();
    expect(why('prl1pexample0123456789abcdef.rig7')).toBeNull();
    expect(why('')).toBe('Paste your Pearl address first.');
    const kryptex = 'Kryptex accounts stop working in the next release. Use your prl1p… Pearl address instead.';
    expect(why('krxX7D2W4E')).toBe(kryptex);
    expect(why('krxX7D2W4E.rig1')).toBe(kryptex);
    expect(why('miner@example.com')).toBe(kryptex);
    expect(why('hello')).toBe("That doesn't look like a Pearl address. It starts with prl1p.");
    // The hint is a live region, so a screen reader says it too.
    expect(page).toContain('<div class="fs-hint" id="fs-hint" role="status" hidden>');
  });

  // The flight sheet the Copy button puts on the clipboard, built by the page's
  // own function, reading the same fields the manual table shows.
  function flightSheet(page, sec, address) {
    const src = page.match(/function flightSheet\(address\) \{[\s\S]*?\n {4}\}/)[0];
    const document = { getElementById: (id) => ({ textContent: codeText(sec, id) }) };
    return JSON.parse(vm.runInNewContext(src + '; flightSheet(address)', { document, address }));
  }

  // HiveOS's Import from Clipboard takes {name, isFavorite, items}. It can't
  // carry a HiveOS wallet, so the address goes in the wallet template, which
  // h-config.sh reads as the address and anything after a '.' as the worker.
  it('builds a flight sheet for the custom miner with the address in the wallet template', async () => {
    const { page, sec } = await section();
    const sheet = flightSheet(page, sec, 'prl1pexample.rig7');
    expect(sheet.items).toHaveLength(1);
    const item = sheet.items[0];
    expect(item.miner).toBe('custom');
    expect(item.coin).toBe('PRL');
    expect(item.miner_config).toMatchObject({
      miner: minerName,
      install_url: URL,
      template: 'prl1pexample.rig7',
      url: 'us.pearl.herominers.com:1200',
    });
    const config = fs.readFileSync(path.join(ROOT, 'earn/hiveos/h-config.sh'), 'utf8');
    expect(config).toContain('wallet=${template%%.*}');
    // The steps say HiveOS still asks for a wallet and that the miner already
    // mines to the address, give the file import for when HiveOS can't read the
    // clipboard, link the LLM guide as an optional last step, and the button and
    // the file link are there.
    expect(sec).toContain('Import from Clipboard');
    expect(sec).toContain('Import from File');
    expect(sec).toContain('HiveOS asks for a <b>Wallet</b>');
    expect(sec).toContain('Pick your PRL wallet');
    expect(sec).toContain('already mines to your address');
    expect(sec).toMatch(/<span class="n">4<\/span><span>Optional: [^<]*<a class="hive-llm" href="\/hiveos-llm">/);
    expect(sec).toContain('id="fs-copy"');
    expect(sec).toContain('id="fs-file"');
  });

  // HiveOS reads which cards are off only when the miner starts, and only the
  // mining follows it, so the page must not say more than that.
  it('says what turning a card off does and does not do', async () => {
    const { sec } = await section();
    expect(sec).toContain("A card you turn off in HiveOS doesn't mine.");
    expect(sec).toContain('restart the miner after turning a card on or off');
    expect(sec).toContain("The local LLM from <code>--mode auto</code> doesn't follow it.");
  });

  // Umami counts the HiveOS path like the other platforms. A Copy of the
  // Installation URL is the HiveOS download; the other fields' Copy buttons are
  // not. Opening the flight sheet from a menu is its own event, so the two
  // aren't counted as two downloads.
  it('is linked from both download menus, and Umami counts its links and the URL Copy', async () => {
    const { page, sec } = await section();
    const links = page.match(/<a role="menuitem"[^>]*href="#hiveos"[^>]*>/g);
    expect(links).toHaveLength(2);
    for (const a of links) expect(a).toContain('data-umami-event="hiveos-flight-sheet"');
    expect(links[0]).toContain('data-umami-event-place="top"');
    expect(links[1]).toContain('data-umami-event-place="bottom"');
    const copies = sec.match(/<button class="hive-copy"[^>]*>/g);
    const counted = copies.filter((b) => b.includes('data-umami-event='));
    expect(counted).toHaveLength(1);
    expect(counted[0]).toContain('data-copy="hive-url"');
    expect(counted[0]).toContain('data-umami-event="download"');
    expect(counted[0]).toContain('data-umami-event-os="hiveos"');
    expect(counted[0]).toContain('data-umami-event-place="flight-sheet"');
    // A flight sheet copied or saved is a HiveOS download too. The script counts
    // it only once there was an address to put in it.
    expect(page).toContain("umami.track('download', { os: 'hiveos', place: place })");
    expect(page).toContain("count('flight-sheet-copy')");
    expect(page).toContain("count('flight-sheet-file')");
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
