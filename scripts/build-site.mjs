#!/usr/bin/env node
// Static-site builder for the marketing/dashboard pages served from the repo
// root (by GitHub Pages and by the Express static handler). Pages live in
// site/pages/ and pull shared chrome from site/partials/ and shared constants
// from site/config.json, so the nav, footer, <head> boilerplate, favicon, API
// base and release version are defined once instead of copy-pasted into five
// files (where they had already drifted).
//
// Templating (deliberately tiny — no dependencies):
//   {{> name}}          include site/partials/name.html (recursively rendered)
//   {{#flag}}...{{/flag}}  keep the body only when the flag is truthy
//   {{key}}             substitute a value from config + the page's front-matter
//
// A page declares its front-matter as a JSON comment on the first line:
//   <!--build {"title":"…","active":"earn","clerk":true} -->
//
// Output is written to the repo root (the current served location), so this
// changes no deploy paths. `--check` builds to memory and fails if the committed
// output is stale, which is what keeps the generated files honest in CI.

import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SITE = join(ROOT, 'site');
const PARTIALS = join(SITE, 'partials');
const PAGES = join(SITE, 'pages');

const config = JSON.parse(readFileSync(join(SITE, 'config.json'), 'utf8'));

const partialCache = new Map();
function partial(name) {
  if (!partialCache.has(name)) {
    partialCache.set(name, readFileSync(join(PARTIALS, name + '.html'), 'utf8'));
  }
  return partialCache.get(name);
}

// Pull the optional first-line JSON front-matter off a page.
function parseFrontMatter(src) {
  const m = src.match(/^\s*<!--build\s+(\{[\s\S]*?\})\s*-->\s*\n?/);
  if (!m) return { data: {}, body: src };
  return { data: JSON.parse(m[1]), body: src.slice(m[0].length) };
}

function render(src, ctx, depth = 0) {
  if (depth > 20) throw new Error('template include depth exceeded (cycle?)');
  let out = src;
  // Includes first, so partials can themselves use sections/variables.
  out = out.replace(/\{\{>\s*([\w-]+)\s*\}\}/g, (_, name) => render(partial(name), ctx, depth + 1));
  // Sections: keep the body only when the flag is truthy.
  out = out.replace(/\{\{#(\w+)\}\}([\s\S]*?)\{\{\/\1\}\}/g, (_, key, body) => (ctx[key] ? body : ''));
  // Variables. Only known keys are substituted; anything else is left verbatim
  // so React inline styles in a page body (`style={{ ... }}`) can't collide with
  // the template syntax. `{{!key}}` asserts a key must exist — use it in
  // partials to still catch typos in the pieces we control.
  out = out.replace(/\{\{!\s*([\w.]+)\s*\}\}/g, (whole, key) => {
    if (Object.prototype.hasOwnProperty.call(ctx, key)) return String(ctx[key]);
    throw new Error('unknown required template variable: ' + whole);
  });
  out = out.replace(/\{\{\s*([\w.]+)\s*\}\}/g, (whole, key) =>
    (Object.prototype.hasOwnProperty.call(ctx, key) ? String(ctx[key]) : whole));
  return out;
}

function buildPage(file) {
  const { data, body } = parseFrontMatter(readFileSync(join(PAGES, file), 'utf8'));
  const ctx = { ...config, ...data };
  return render(body, ctx);
}

const pages = readdirSync(PAGES).filter((f) => f.endsWith('.html'));
const check = process.argv.includes('--check');
let stale = 0;

for (const file of pages) {
  const rendered = buildPage(file);
  const dest = join(ROOT, file);
  if (check) {
    const current = readFileSync(dest, 'utf8');
    if (current !== rendered) {
      stale++;
      console.error('stale: ' + file + ' — run `npm run build:site`');
    }
  } else {
    writeFileSync(dest, rendered);
    console.log('built ' + file);
  }
}

if (check && stale) {
  console.error('\n' + stale + ' page(s) out of date with site/ sources.');
  process.exit(1);
}
if (check) console.log('all ' + pages.length + ' page(s) up to date');
