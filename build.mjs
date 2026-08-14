#!/usr/bin/env node
/**
 * Build the docs site as ordinary static HTML.
 *
 * Each page's real content is the `guideHtml` template literal inside its
 * `.mdx` — the Mintlify wrapper (`mode: "custom"`, an HTML string in JS) exists
 * only because Mintlify routes `.md`/`.mdx` and nothing else. Served as plain
 * files, that wrapper is unnecessary; this script unwraps it.
 *
 * WHY `<body id="lloyal-guides" class="pg-<slug>">`: the stylesheet is scoped
 * per page as `#lloyal-guides.pg-<slug>` — added deliberately to stop one
 * page's rules bleeding into another. Reproducing both on the body means all
 * 278 rules apply unchanged, so this migration is provably a hosting change
 * with no visual diff. The names are vestigial; flattening them is a separate
 * change with its own before/after comparison.
 *
 * Source of truth is the `.mdx` + `assets/guides/guides-theme.css` in this
 * repo — NOT ~/Downloads/lloyal-programming-guides, which predates the subnav
 * edits and the Reference page.
 */
import { readFileSync, writeFileSync, mkdirSync, rmSync, cpSync, readdirSync, existsSync, renameSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, basename } from 'node:path';

const OUT = 'dist';

/**
 * The stylesheet is content-hashed. The images are content-stable, so serving
 * them `immutable` is safe — the CSS is NOT: it changes with every style edit,
 * and an immutable un-hashed filename means those edits can never reach anyone
 * (found the hard way: a fix deployed, and the CDN kept serving the old file).
 * Hashing the name makes `immutable` correct rather than a trap.
 */
const cssBody = readFileSync('assets/guides/guides-theme.css');
const CSS_NAME = `guides-theme.${createHash('sha256').update(cssBody).digest('hex').slice(0, 8)}.css`;
const CSS = `/assets/guides/${CSS_NAME}`;

/** Frontmatter `title` / `description`, which become the <head>. */
function frontmatter(src) {
  const fm = /^---\n([\s\S]*?)\n---/.exec(src)?.[1] ?? '';
  const get = (k) => /^\s*$/.test(fm) ? '' : (new RegExp(`^${k}:\\s*"?(.*?)"?\\s*$`, 'm').exec(fm)?.[1] ?? '');
  return { title: get('title'), description: get('description') };
}

/**
 * The page body. `guideHtml` is a template literal, so it ends at the first
 * unescaped backtick — every page is verified to contain none inside, and an
 * escaped one (\`) is unescaped back to a literal backtick.
 */
function guideHtml(src) {
  const start = src.indexOf('export const guideHtml = `');
  if (start < 0) return null;
  const from = start + 'export const guideHtml = `'.length;
  let out = '';
  for (let i = from; i < src.length; i++) {
    if (src[i] === '\\' && src[i + 1] === '`') { out += '`'; i++; continue; }
    if (src[i] === '`') return out;
    out += src[i];
  }
  throw new Error('unterminated guideHtml literal');
}

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

/**
 * Highlight the section you are currently reading in the sidebar TOC.
 *
 * The TOC is already `position: sticky` in the stylesheet, so it stays put on
 * its own — but with 40+ entries on the longer guides there was no indication
 * of where you were in the page. This adds that and nothing else.
 *
 * Deliberately small: no dependencies, no framework, one IntersectionObserver,
 * and it degrades to the current behaviour if JS is off. The original export
 * shipped zero JavaScript and this is the only script on the site.
 */
/**
 * The menu ships on every page. It lived inside TOC_SCRIPT, which is only
 * injected when a page has a table of contents — so the button rendered on the
 * hub and the 404 page with nothing listening to it.
 */
const NAV_SCRIPT = `<script>
/* Guides menu.

   Below the nav breakpoint the bar collapses to a button. Toggling an attribute
   on the nav is the whole of it — the open and closed layouts are both CSS, so
   there is no measuring here and nothing to keep in sync. */
(function () {
  var nav = document.querySelector('.lloyal-topnav');
  if (!nav) return;
  var btn = nav.querySelector('.lloyal-navtoggle');
  if (!btn) return;
  btn.addEventListener('click', function () {
    var open = nav.hasAttribute('data-open');
    if (open) nav.removeAttribute('data-open');
    else nav.setAttribute('data-open', '');
    btn.setAttribute('aria-expanded', String(!open));
  });
  // A link press navigates; leaving the panel open would flash it on the next page.
  nav.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') nav.removeAttribute('data-open');
  });
  addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && nav.hasAttribute('data-open')) {
      nav.removeAttribute('data-open');
      btn.setAttribute('aria-expanded', 'false');
      btn.focus();
    }
  });
})();
</script>`;

const TOC_SCRIPT = `<script>
(function () {
  var toc = document.querySelector('.toc');
  if (!toc) return;
  var links = [].slice.call(toc.querySelectorAll('a[href^="#"]'));
  var items = links.map(function (a) {
    return { a: a, el: document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))) };
  }).filter(function (i) { return i.el; });
  if (!items.length) return;

  var current = null;

  function sync() {
    // The current section is simply the last heading whose top has passed the
    // read line. A plain geometric test rather than IntersectionObserver: with
    // 68 entries the observer only fires for headings crossing a thin band, so
    // it never settles on the right one after a jump or on load.
    var line = window.innerHeight * 0.25, found = items[0];
    for (var i = 0; i < items.length; i++) {
      if (items[i].el.getBoundingClientRect().top <= line) found = items[i];
      else break;
    }
    if (found === current) return;
    if (current) current.a.classList.remove('active');
    current = found;
    found.a.classList.add('active');

    // Keep the active entry inside the TOC's own scroll box, so the sidebar
    // follows the page and never has to be scrolled by hand. Adjust scrollTop
    // directly — scrollIntoView() would scroll the document as well.
    var box = toc.getBoundingClientRect(), row = found.a.getBoundingClientRect();
    if (row.top < box.top + 40) toc.scrollTop -= (box.top + 40 - row.top);
    else if (row.bottom > box.bottom - 40) toc.scrollTop += (row.bottom - box.bottom + 40);
  }

  var queued = false;
  function onScroll() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; sync(); });
  }
  addEventListener('scroll', onScroll, { passive: true });
  addEventListener('resize', onScroll, { passive: true });
  addEventListener('hashchange', onScroll);
  sync();
})();

/* Back to top.

   Every page already ships an <a class="back-to-top"> and the stylesheet gives
   it opacity:0 until .show — but nothing ever added that class, so the control
   has been inert on all of them. Its own block rather than part of the TOC's:
   that one returns early when a page has no TOC entries. */
(function () {
  var btn = document.querySelector('.back-to-top');
  if (!btn) return;
  function toggle() { btn.classList.toggle('show', window.scrollY > 900); }
  addEventListener('scroll', toggle, { passive: true });
  toggle();
})();
</script>`;

/**
 * Site navigation.
 *
 * The docs are a hub and six spokes, so without this bar the only way from one
 * guide to another is back through the index. The labels are the short names
 * the hub already uses in its learning-flow and path-labels — NOT the page
 * `<h1>`s, which do not fit on one line ("How Agents Know When to Stop"). Six
 * long labels wrapping to two lines is what sank the earlier attempt at this.
 *
 * Built here rather than written into each `.mdx` so the six pages cannot drift
 * apart, and so `current` follows the slug by construction.
 */
const NAV = [
  { slug: 'index', href: '/', label: 'Docs' },
  { slug: 'build-your-first-harness', href: '/build-your-first-harness', label: 'Build your first harness' },
  { slug: 'thinking-in-lloyal', href: '/thinking-in-lloyal', label: 'Thinking in Lloyal' },
  { slug: 'continuous-context', href: '/continuous-context', label: 'Continuous Context' },
  { slug: 'agent-policy-and-context-pressure', href: '/agent-policy-and-context-pressure', label: 'Context pressure' },
  { slug: 'where-a-harness-runs', href: '/where-a-harness-runs', label: 'Where a harness runs' },
  { slug: 'lookup', href: '/lookup', label: 'Lookup' },
];

/**
 * Seven labels need ~1250px. Above that the bar is a row; below it the row
 * becomes a disclosure, because a strip that scrolls sideways shows a phone two
 * entries and gives no sign the other five exist.
 *
 * The button ships in the markup and is hidden by CSS above the breakpoint, so
 * there is nothing to construct at runtime and nothing moves if the script
 * fails; the links are plain anchors either way.
 */
const nav = (slug) => `<nav class="lloyal-topnav" aria-label="Guides">` +
  `<button type="button" class="lloyal-navtoggle" aria-expanded="false" aria-label="Guides menu">` +
  `<span class="lloyal-navtoggle-bars" aria-hidden="true"></span></button>` +
  NAV.map(
    (n) => `<a href="${n.href}"${n.slug === slug ? ' class="lloyal-topnav-current" aria-current="page"' : ''}>${n.label}</a>`,
  ).join('') + `</nav>`;

const pages = readdirSync('.').filter((f) => f.endsWith('.mdx')).sort();
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const built = [];
for (const file of pages) {
  const src = readFileSync(file, 'utf8');
  const body = guideHtml(src);
  if (body === null) { console.log(`  skip   ${file} (no guideHtml)`); continue; }
  const slug = basename(file, '.mdx');
  const { title, description } = frontmatter(src);

  writeFileSync(join(OUT, `${slug}.html`), `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)} — Lloyal Labs</title>
<meta name="description" content="${esc(description)}">
<link rel="icon" href="/favicon.svg">
<link rel="stylesheet" href="${CSS}">
</head>
<body id="lloyal-guides" class="pg-${slug}">
${nav(slug)}
${body}
${NAV_SCRIPT}
${body.includes('class="toc"') ? TOC_SCRIPT : ''}
</body>
</html>
`);
  built.push({ slug, title, description });
  console.log(`  build  ${slug}.html`);
}

// Static assets, copied verbatim — the PNGs are byte-identical to the originals
// and are content-stable, which is what makes the immutable _headers rule safe.
for (const dir of ['assets', 'logo']) if (existsSync(dir)) cpSync(dir, join(OUT, dir), { recursive: true });
// Publish the stylesheet under its content-hashed name only, so a stale copy
// can never be served from an immutable cache entry.
renameSync(join(OUT, 'assets/guides/guides-theme.css'), join(OUT, 'assets/guides', CSS_NAME));
for (const f of ['favicon.svg']) if (existsSync(f)) cpSync(f, join(OUT, f));

// Redirects: every retired URL, carried over from the Mintlify config so no
// previously-working link breaks in the cutover.
const redirects = JSON.parse(readFileSync('docs.json', 'utf8')).redirects ?? [];
writeFileSync(join(OUT, '_redirects'), redirects.map((r) => `${r.source} ${r.destination} 301`).join('\n') + '\n');

// Long-lived caching for assets — the point of extracting the images out of
// base64 in the first place. HTML stays revalidated so edits appear at once.
writeFileSync(join(OUT, '_headers'), `/assets/*
  Cache-Control: public, max-age=31536000, immutable

/*.html
  Cache-Control: public, max-age=0, must-revalidate
`);

// Without a 404.html, Cloudflare Pages falls back to index.html for ANY
// unmatched path — so every dead link returns the homepage with a 200. That is
// a soft 404: links look alive, and crawlers index endless duplicates of the
// hub. This makes a miss say so, in the site's own idiom.
writeFileSync(join(OUT, '404.html'), `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not found — Lloyal Labs</title>
<meta name="robots" content="noindex">
<link rel="icon" href="/favicon.svg">
<link rel="stylesheet" href="${CSS}">
</head>
<body id="lloyal-guides" class="pg-index">
${nav('')}
<div class="page">
<header class="masthead wrap">
<div><div class="wordmark">Lloyal Labs</div><div class="tagline">Engineering AI's contact with reality.</div></div>
<div class="issue"><span>Not /<br>found</span></div>
</header>
<section class="hero wrap">
<div class="eyebrow"><a href="/">Programming guides</a> / 404</div>
<h1>Not found</h1>
<div class="hero-sub">That page does not exist here.</div>
</section>
<section class="references wrap">
<div class="section-kicker">Try instead</div><h2>The guides</h2>
<div class="reference-grid">
<a href="/">Build with Lloyal <span class="out">&rarr;</span></a>
<a href="/build-your-first-harness">Build your first harness <span class="out">&rarr;</span></a>
<a href="/thinking-in-lloyal">Thinking in Lloyal <span class="out">&rarr;</span></a>
<a href="/continuous-context">Continuous Context <span class="out">&rarr;</span></a>
<a href="/agent-policy-and-context-pressure">Context pressure <span class="out">&rarr;</span></a>
<a href="/where-a-harness-runs">Where a harness runs <span class="out">&rarr;</span></a>
<a href="/lookup">Lookup <span class="out">&rarr;</span></a>
</div>
</section>
<footer class="footer wrap"><span class="mission">Enable every organisation to deliver intelligence on its terms.</span><span>Lloyal Labs &middot; Melbourne</span></footer>
</div>
${NAV_SCRIPT}
</body>
</html>
`);

// Mintlify generated this; emit an equivalent so it does not vanish silently.
writeFileSync(join(OUT, 'llms.txt'), `# Lloyal HDK documentation

${built.map((p) => `- [${p.title}](https://docs.lloyal.ai/${p.slug === 'index' ? '' : p.slug}): ${p.description}`).join('\n')}
`);

console.log(`\n  ${built.length} pages · ${redirects.length} redirects · ${CSS_NAME} → ${OUT}/`);
