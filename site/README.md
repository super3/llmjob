# Static site sources

The marketing/dashboard pages (`index.html`, `llm.html`, `network.html`,
`docs.html`, `terms.html`, `privacy.html`, `add-node.html`, `dashboard.html`)
are
**generated** from the sources here into
`dist/` by `site/build-site.mjs`. `dist/` is git-ignored — nothing generated
is committed. Edit the sources in this directory, never a built page.

## URLs

Pages are served **without the `.html`**: `network.html` is `/network`, and the home
page is `/`. The source and built files keep their extension — only the URL
drops it, and both hosts resolve it:

- **GitHub Pages** strips the extension itself; nothing to configure.
- **The Express server** passes `extensions: ['html']` to `express.static`, and
  301-redirects `/network.html` → `/network` (and `/index.html` → `/`) so a page never
  answers on two URLs at once.

So link between pages as `href="/network"` — root-absolute, no extension — and
write `og:url` and any `window.location` / Clerk redirect the same way. A link
that still carries `.html` works, but costs the visitor a redirect.

### Retired pages

When a page is removed or moved, add its old name to `site/redirects.json`
(`{ "chat": "/" }`) so old links still land somewhere. `/earn` is there too:
the Earn download page is now the home page (`index.html`), and the "Run LLMs
at home" page that used to be the home page is `llm.html`. The build writes a small `dist/<name>.html`
that redirects the visitor to the target. It works on both hosts because it is
just a page, and the build fails if a redirect has the same name as a real page.

## Where the output goes

- **GitHub Pages** builds `dist/` in `.github/workflows/deploy.yml` and publishes
  it as the Pages artifact (only `dist/` — the source tree is no longer served).
- **The Express server** builds `dist/` on start (`npm start`) and serves it, so
  the Railway deployment answers for the same pages.

## Layout

- `build-site.mjs` — the builder itself (plain Node, no dependencies). It lives
  next to the sources it renders; `dist/` is written to the repo root.
- `pages/` — one source file per page. Each may start with a JSON front-matter
  comment (`<!--build { … } -->`) declaring page variables (`clerk`, `fonts`, …).
- `partials/` — shared fragments pulled in with `{{> name}}`:
  - `head.html` — analytics, Clerk loader (when `clerk` is set), favicon, fonts.
  - `api-base.html` — the shared `API_BASE` origin resolution.
- `config.json` — shared constants (analytics id, Clerk key, API host, release
  version, …) available to every page and partial as `{{key}}`.
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
