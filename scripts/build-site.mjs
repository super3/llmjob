#!/usr/bin/env node
// Static-site builder for the marketing/dashboard pages. Sources live in
// site/pages/ and pull shared chrome from site/partials/ and shared constants
// from site/config.json, so the <head> boilerplate (analytics, Clerk, favicon,
// fonts), the API-base resolution and the release version are defined once
// instead of copy-pasted into five files (where they had already drifted).
//
// Output goes to dist/ (git-ignored, not committed). GitHub Pages builds this
// in its workflow and publishes dist/ as the artifact; the Express server
// builds it on start and serves dist/. No generated file is ever committed, so
// there's nothing to keep "fresh" — the sources in site/ are the only truth.
//
// Templating (deliberately tiny — no dependencies):
//   {{> name}}             include site/partials/name.html (recursively rendered)
//   {{#flag}}...{{/flag}}  keep the body only when the flag is truthy
//   {{key}}                substitute a value from config + the page's front-matter
//                          (unknown keys are left verbatim, so a page body's
//                           React `style={{ ... }}` can't collide with the syntax)
//   {{!key}}               like {{key}} but a build error if the key is missing —
//                          use in partials to catch typos in the shared pieces
//
// A page declares optional front-matter as a JSON comment on the first line:
//   <!--build {"active":"earn","clerk":true,"fonts":"…"} -->

import { readFileSync, writeFileSync, readdirSync, mkdirSync, rmSync, cpSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SITE = join(ROOT, 'site');
const PARTIALS = join(SITE, 'partials');
const PAGES = join(SITE, 'pages');
const STATIC = join(SITE, 'static'); // optional passthrough assets (images, etc.)
const OUT = join(ROOT, 'dist');

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
  // Required variables (assert), then optional variables (leave unknowns as-is).
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

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

// Static passthrough (only if site/static/ exists), then the rendered pages.
if (existsSync(STATIC)) {
  cpSync(STATIC, OUT, { recursive: true });
}

const pages = readdirSync(PAGES).filter((f) => f.endsWith('.html'));
for (const file of pages) {
  writeFileSync(join(OUT, file), buildPage(file));
  console.log('built dist/' + file);
}
console.log('built ' + pages.length + ' page(s) → ' + OUT);
