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
domain in `CNAME`. There is no deploy workflow; GitHub builds it directly with
the `github-pages` gem (Jekyll 3.10), not the Jekyll 4 in `Gemfile.lock`.
`.github/workflows/jekyll.yml` is a **build check only** and never deploys: it
builds with `actions/jekyll-build-pages`, checks key pages exist, then runs
htmlproofer 4.4.3 on internal links, image alt text and scripts. (htmlproofer
5.x silently checked nothing when tried; keep the pin.)

To confirm what is actually being served before assuming:

```bash
gh api repos/MurtoHilali/murtohilali.github.io/pages --jq '{source,build_type,cname}'
```

## Where content lives

| What | Where | Rendered on |
| --- | --- | --- |
| Projects | `_data/projects.yml` (the only place a project is defined) | homepage "Selected work" (`featured: true`, sorted by `featured_order`), `/lab/` (grouped by `track`), About ("currently working on" = `status: active`), `llms.txt` |
| Project pixel icons | `_data/icons.yml` (16×16 grids, rendered to SVG by `_includes/pixel-icon.html`) | homepage tiles, Lab rows |
| Publications | `_data/publications.yml` | homepage, About, `llms.txt` |
| Profile links | `_data/social.yml` | hero (`hero: true`), footer, homepage Person JSON-LD |
| Posts | `_posts/`, permalink `/:slug/` | `/blog/` (Writing), homepage |
| Resume | `resume.pdf` → `/resume.pdf` | header "CV", hero, About |

The schema for a project is documented at the top of `_data/projects.yml`.
Links to posts on this site use `{ label, post: <slug> }` rather than a URL.

### Posts: the `kind` field

Every post has `kind: writeup` (long-form technical write-up) or `kind: news`
(short update). That field, not tags, decides where a post is listed: the
homepage "Writing" list is the 5 newest write-ups, "News" the 5 newest news
posts, and `/blog/` filters by it (`/blog/#writeups`, `/blog/#news`). Tags are
topical only (`bioinformatics`, `ml`, ...) and feed search. Don't use a
`project` or `post` tag to drive a listing.

Post bodies (below the front matter) are left as written. The post layout adds
image loading hints at render time instead of editing them.

## Layout notes

- One header for every page: `_includes/site-header.html` (name → home, Lab,
  Writing, About, CV, theme toggle). `_layouts/default.html` renders it; so
  does `_layouts/lab.html`. Footer is `_includes/footer.html`.
- Styles are SCSS partials inlined into each page via `_includes/head.html`:
  `_fonts` (self-hosted woff2 in `assets/google_fonts/`), `_tokens` (light/dark
  custom properties), `_chrome` (header, toggle, footer, focus), then `_main` +
  `_home` for the site, or `_lab` for the Lab.
- **The Lab** (`lab.html`, served at `/lab/`) uses `layout: lab`
  (`_layouts/lab.html`). Rows are rendered at build time from
  `_data/projects.yml` via `_includes/project-row.html` as native
  `<details>`, so the page works without JS; `/lab/#<project id>` opens a row.
  Its CSS is `_sass/_lab.scss`, scoped under `body.lab` and loaded through
  `_includes/lab-inline.scss` (which skips `_main.scss`). `assets/js/lab.js`
  only adds extras (status clock, ASCII monitors, ticker, sound, which is off by
  default and creates no AudioContext until enabled). There is no Tailwind and
  no inline `PROJECTS` array any more.
- `travels.html` is still a **standalone page**: a full HTML document with
  `permalink: /travels/` and `layout: null`. `layout: null` is load-bearing,
  because `_config.yml` defaults every page to `default`. It doesn't use the
  site header or theme.
- The `compress` layout joins lines in inline `<script>`s, so inline JS must not
  use `//` comments. Larger scripts live in `assets/js/`.
- `llms.txt` is generated at build time (Projects from data, Write-ups, News,
  Publications). Edit the Liquid in `llms.txt`, not the output.
- Redirects use `jekyll-redirect-from`. `/cv/` and the old resume URLs are stubs
  in `redirects/`. An old URL ending in `.pdf` is a directory stub (`<name>.pdf/`)
  because GitHub Pages serves anything named `.pdf` as a PDF and 301s the
  slashless path to the directory.
- Records (`murto.co/records/`) is a separate repo (`MurtoHilali/records`)
  served as a project site; `records.murto.co` doesn't work.

## Conventions

- Don't commit notebooks, datasets, or other scratch files — the site repo is
  for the site. `notes/` is gitignored for local scratch.
- Verify external links resolve before committing them.
- Build with `bundle exec jekyll build`; `bundle exec` may rewrite the platform
  lines in `Gemfile.lock` on macOS. Don't commit that churn.
