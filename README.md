# murto.co

Personal site of Murto Hilali: computational biology and protein design
projects, write-ups, and news. Built with Jekyll and served by GitHub Pages
at **https://murto.co**.

## Run it locally

```bash
bundle install
bundle exec jekyll serve      # http://localhost:4000
```

`bundle exec jekyll build` writes the site to `_site/`.

## Where things live

| What | Where |
| --- | --- |
| Projects (homepage "Selected work", the Lab, About, `llms.txt`) | `_data/projects.yml` |
| Project pixel icons | `_data/icons.yml` |
| Publications (homepage, About, `llms.txt`) | `_data/publications.yml` |
| Profile links (hero, footer) | `_data/social.yml` |
| Write-ups and news | `_posts/`, with `kind: writeup` or `kind: news` |
| The Lab | `lab.html` + `_layouts/lab.html`, styles in `_sass/_lab.scss`, extras in `assets/js/lab.js` |
| Resume | `resume.pdf`, served at `/resume.pdf` (`/cv/` redirects to it) |

To add a project, add an entry to `_data/projects.yml`; set `featured: true`
and a `featured_order` to put it on the homepage. To add a post, put it in
`_posts/` with a `kind`.

## Branches

`main` is served: GitHub Pages builds it directly (legacy build, custom
domain in `CNAME`), so a push to `main` is a deploy. Work happens on
`development` and is merged into `main` to publish. The workflow in
`.github/workflows/jekyll.yml` only checks that the site builds and that
internal links resolve; it never deploys.

See `CLAUDE.md` for more detail on the layout.
