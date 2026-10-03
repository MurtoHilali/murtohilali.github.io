# murtohilali.github.io

Jekyll site served at **murto.co**.

## Branches

| Branch | Role |
| --- | --- |
| `main` | **Served.** GitHub Pages builds the live site from here. Default branch. |
| `development` | Working branch for feature work. Not served. |

Work on `development`, then merge into `main` to publish. A push to `main` is a
deploy — treat it that way.

Pages is configured as a **legacy** build from `main` at `/`, with the custom
domain in `CNAME`. There is no deploy workflow; GitHub builds it directly.
`.github/workflows/jekyll.yml` is a **build check only** and never deploys.

To confirm what is actually being served before assuming:

```bash
gh api repos/MurtoHilali/murtohilali.github.io/pages --jq '{source,build_type,cname}'
```

## Layout notes

- Posts live in `_posts/`, permalink is `/:slug/` — so
  `_posts/2024-05-04-xgboost-ppi.md` is served at `murto.co/xgboost-ppi`.
- A post appears on the homepage under **Projects** if its `tags` contain
  `project`, and under **Posts** if they contain `post`.
- `lab.html` and `travels.html` are **standalone pages**: full HTML documents
  that do not use `_layouts` and are unaffected by the site's theme switcher.
  Each carries only `permalink:` and `layout: null` in its front matter —
  `layout: null` is load-bearing, because `_config.yml` defaults every processed
  page to the `default` layout, which would wrap and break them. They are served
  at `/lab/` and `/travels/`. `lab.html`'s project data is the `PROJECTS` array
  in its inline script.
- `llms.txt` is generated from the posts at build time, so it stays current.
  Don't hand-edit the output — edit the Liquid in `llms.txt`.
- `index_og.html` and `index_test.html` are old homepage drafts. They *are*
  still built and reachable at `/index_og` and `/index_test` — two extra
  copies of the homepage. Worth excluding if you don't want them indexed.

## Conventions

- Don't commit notebooks, datasets, or other scratch files — the site repo is
  for the site.
- Verify external links resolve before committing them.
