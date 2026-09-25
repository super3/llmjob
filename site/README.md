# Static site sources

The site's pages (`index.html` — the download page and earnings calculator,
`network.html`, `managed.html`, `terms.html`, `privacy.html`) are
**generated** from the sources here into
`dist/` by `site/build-site.mjs`. `dist/` is git-ignored — nothing generated
is committed. Edit the sources in this directory, never a built page.

## URLs

Pages are served **without the `.html`**: `network.html` is `/network`, and the
home page is `/`. The source and built files keep their extension — only the URL
drops it, and both hosts resolve it:

- **GitHub Pages** strips the extension itself; nothing to configure.
- **The Express server** passes `extensions: ['html']` to `express.static`, and
  301-redirects `/network.html` → `/network` (and `/index.html` → `/`) so a page
  never answers on two URLs at once.

So link between pages as `href="/network"` — root-absolute, no extension — and
write `og:url` and any `window.location` redirect the same way.

## Retired pages

`redirects.json` maps a retired page to where it lives now (`"earn": "/"`). The
builder writes a small stub for each — `dist/earn.html` — that sends the visitor
on, keeping any `#fragment`, so links already out in the world (videos, Discord
posts, bookmarks, older app versions) don't 404. A stub rather than a server
redirect because GitHub Pages can't do redirects. The build fails if a redirect
would shadow a real page. A link
that still carries `.html` works, but costs the visitor a redirect.

## Where the output goes

- **GitHub Pages** builds `dist/` in `.github/workflows/deploy.yml` and publishes
  it as the Pages artifact (only `dist/` — the source tree is no longer served).
- **The Express server** builds `dist/` on start (`npm start`) and serves it, so
  the Railway deployment answers for the same pages.

## Layout

- `build-site.mjs` — the builder itself (plain Node, no dependencies). It lives
  next to the sources it renders; `dist/` is written to the repo root.
- `pages/` — one source file per page. Each may start with a JSON front-matter
  comment (`<!--build { … } -->`) declaring page variables (`navHome`, `fonts`, …).
- `partials/` — shared fragments pulled in with `{{> name}}`:
  - `head.html` — analytics, favicon, fonts.
  - `api-base.html` — the shared `API_BASE` origin resolution.
- `config.json` — shared constants (analytics id, API host, release version, …)
  available to every page and partial as `{{key}}`.
- `redirects.json` — retired page → new URL (see above).
- `static/` — optional; anything here is copied verbatim into `dist/` (images,
  etc.). Does not exist yet — the pages are currently self-contained.

## Templating

- `{{> name}}` — include `partials/name.html` (rendered recursively).
- `{{#flag}}…{{/flag}}` — keep the body only when `flag` is truthy.
- `{{key}}` — substitute a value; unknown keys are left as-is (so React
  `style={{…}}` in a page body is safe).
- `{{!key}}` — like `{{key}}`, but a build error if the key is missing. Use it in
  partials to catch typos in the pieces we control.

## Build / preview

```sh
npm run build:site           # render site/ → dist/
npx serve dist               # preview locally (any static server works)
```
