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
- `lab.html` is a **standalone page** with no front matter. Jekyll copies it
  through verbatim, so it does not use `_layouts` and is unaffected by the
  site's theme switcher. Its project data is the `PROJECTS` array in the inline
  script.
- `index_og.html` and `index_test.html` are old homepage drafts. They *are*
  still built and reachable at `/index_og` and `/index_test` — two extra
  copies of the homepage. Worth excluding if you don't want them indexed.

## Conventions

- Don't commit notebooks, datasets, or other scratch files — the site repo is
  for the site.
- Verify external links resolve before committing them.
