# Static site sources

The marketing/dashboard pages served from the repo root (`index.html`,
`earn.html`, `network.html`, `add-node.html`, `dashboard.html`) are **generated**
from the sources in this directory. Don't edit the root `*.html` files directly —
CI (`npm run build:site:check`) will fail if they drift from these sources.

## Layout

- `pages/` — one source file per page. Each may start with a JSON front-matter
  comment (`<!--build { … } -->`) declaring page variables (`clerk`, `fonts`, …).
- `partials/` — shared fragments pulled in with `{{> name}}`:
  - `head.html` — analytics, Clerk loader (when `clerk` is set), favicon, fonts.
  - `api-base.html` — the shared `API_BASE` origin resolution.
- `config.json` — shared constants (analytics id, Clerk key, API host, release
  version, …) available to every page and partial as `{{key}}`.

## Templating

- `{{> name}}` — include `partials/name.html` (rendered recursively).
- `{{#flag}}…{{/flag}}` — keep the body only when `flag` (config/front-matter) is truthy.
- `{{key}}` — substitute a value; unknown keys are left as-is (so React `style={{…}}`
  in a page body is safe).
- `{{!key}}` — like `{{key}}`, but a build error if the key is missing. Use it in
  partials to catch typos in the pieces we control.

## Build

```sh
npm run build:site        # regenerate the root *.html
npm run build:site:check  # verify the committed output matches (CI)
```

The GitHub Pages workflow rebuilds before deploying, so the published site is
always fresh even if a commit forgets to.
