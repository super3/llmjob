# Static site sources

The marketing/dashboard pages (`index.html`, `earn.html`, `network.html`,
`add-node.html`, `dashboard.html`) are **generated** from the sources here into
`dist/` by `scripts/build-site.mjs`. `dist/` is git-ignored — nothing generated
is committed. Edit the sources in this directory, never a built page.

## Where the output goes

- **GitHub Pages** builds `dist/` in `.github/workflows/deploy.yml` and publishes
  it as the Pages artifact (only `dist/` — the source tree is no longer served).
- **The Express server** builds `dist/` on start (`npm start`) and serves it, so
  the Railway deployment answers for the same pages.

## Layout

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
