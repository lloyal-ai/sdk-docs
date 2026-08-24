#!/usr/bin/env node
/**
 * Build the docs site as ordinary static HTML.
 *
 * Each page is an `.html` file: a metadata comment, then the body markup. It
 * used to be an `.mdx` wrapping that markup in a `guideHtml` template literal,
 * because Mintlify routes `.md`/`.mdx` and nothing else. Mintlify is gone, and
 * the wrapper had stopped being merely redundant: a template literal makes
 * authors escape backtick, dollar and backslash, and getting that wrong shipped
 * `\${query}` verbatim into the starter code sample. A stray backtick was worse
 * still - the literal simply ended there and the rest of the page vanished, with
 * a green build.
 *
 * WHY `<body id="lloyal-guides" class="pg-<slug>">`: the stylesheet is scoped
 * per page as `#lloyal-guides.pg-<slug>` — added deliberately to stop one
 * page's rules bleeding into another. Reproducing both on the body means all
 * 278 rules apply unchanged, so this migration is provably a hosting change
 * with no visual diff. The names are vestigial; flattening them is a separate
 * change with its own before/after comparison.
 *
 * Source of truth is the `.html` + `assets/guides/guides-theme.css` in this
 * repo — NOT ~/Downloads/lloyal-programming-guides, which predates the subnav
 * edits and the Reference page.
 */
import { readFileSync, writeFileSync, mkdirSync, rmSync, cpSync, readdirSync, existsSync, renameSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, basename, dirname } from 'node:path';
import { COHORT, bodyClass } from './cohorts.mjs';

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

/**
 * Page metadata, from the leading HTML comment. Same two keys the YAML front
 * matter carried, so `<title>`, the description meta and `llms.txt` are
 * unchanged by the format change.
 */
function frontmatter(src) {
  const fm = /^\s*<!--\n([\s\S]*?)\n-->/.exec(src)?.[1] ?? '';
  const get = (k) => new RegExp(`^${k}:\\s*(.*?)\\s*$`, 'm').exec(fm)?.[1] ?? '';
  return { title: get('title'), description: get('description') };
}

/** The page body: everything after the metadata comment. */
function pageBody(src) {
  const m = /^\s*<!--\n[\s\S]*?\n-->\n?/.exec(src);
  if (!m) return null;
  return src.slice(m[0].length);
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
 * `<h1>`s, which do not fit on one line ("Adaptive compute through semantic pruning"). Six
 * long labels wrapping to two lines is what sank the earlier attempt at this.
 *
 * Built here rather than written into each page so the six cannot drift
 * apart, and so `current` follows the slug by construction.
 */

/**
 * The "try instead" grid on 404, derived from NAV rather than typed out.
 *
 * Hand-maintained, it had fallen two pages behind - Abilities and Focus shipped
 * and never reached it, so the page telling you where to go omitted two of the
 * places you could go. Deriving it means adding a page cannot leave it stale.
 * Only the home label differs: "Docs" is right in a nav bar, less so as the
 * first thing offered after a dead link.
 */
const notFoundGrid = () => NAV
  .map(({ href, label }) => `<a href="${href}">${href === '/' ? 'Build with Lloyal' : label} <span class="out">&rarr;</span></a>`)
  .join('\n');

// Absolute origins. Canonical and Open Graph URLs must be absolute, and the
// category link is what ties every page to the definition it documents.
const SITE = 'https://docs.lloyal.ai';
const CATEGORY_URL = 'https://verticalinference.lloyal.ai/';
// Shared with the marketing site rather than duplicated here; the card art is
// authored in Figma and lives at one address.
const OG_IMAGE = 'https://lloyal.ai/assets/home-og.png';
const canonical = (slug) => (slug === 'index' ? `${SITE}/` : `${SITE}/${slug}`);

const NAV = [
  { slug: 'index', href: '/', label: 'Docs' },
  { slug: 'build-your-first-harness', href: '/build-your-first-harness', label: 'Build your first harness' },
  { slug: 'thinking-in-lloyal', href: '/thinking-in-lloyal', label: 'Thinking in Lloyal' },
  { slug: 'continuous-context', href: '/continuous-context', label: 'Continuous Context' },
  { slug: 'abilities', href: '/abilities', label: 'Abilities' },
  { slug: 'agent-policy-and-context-pressure', href: '/agent-policy-and-context-pressure', label: 'Adaptive compute' },
  { slug: 'focal-lens', href: '/focal-lens', label: 'Focus' },
  { slug: 'where-a-harness-runs', href: '/where-a-harness-runs', label: 'Where a harness runs' },
  { slug: 'lookup', href: '/lookup', label: 'Lookup' },
];

/**
 * Eight labels need ~1400px. Above that the bar is a row; below it the row
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

/**
 * Every scope the stylesheet targets must be a scope some page actually wears.
 *
 * The two lists live in different files — the markers in `tools/css-gate/merge.mjs`
 * decide what the CSS is keyed on, `COHORT` above decides what the body carries —
 * and nothing but this connects them. Adding `wide` to one and not the other
 * silently deleted the root variables block from all nine pages: no `--max`, so
 * every page went full-bleed. Gate C caught it, but only after a full build and
 * a minute of rendering. This catches it in milliseconds.
 */
function assertScopesExist() {
  const worn = new Set(Object.keys(COHORT).flatMap((s) => bodyClass(s).split(' ')));
  const used = new Set([...readFileSync('assets/guides/guides-theme.css', 'utf8')
    .matchAll(/#lloyal-guides\.([a-z][a-z0-9-]*)/g)].map((m) => m[1]));
  const orphans = [...used].filter((c) => !worn.has(c));
  if (orphans.length) throw new Error(`build: stylesheet targets scopes no page wears: ${orphans.join(', ')}`);
  // And the reverse: a marker every page carries but no rule uses is dead weight
  // in the body class. `wide` became one the moment the two page widths merged.
  const markers = new Set(Object.values(COHORT).flat());
  const unused = [...markers].filter((m) => !used.has(m));
  if (unused.length) throw new Error(`build: COHORT declares scopes the stylesheet never uses: ${unused.join(', ')}`);
}
assertScopesExist();

// One level of nesting, for `licensing/`. Not a general recursive walk - the
// URL surface is small and deliberate, and a page appearing because someone
// dropped a file somewhere is not a feature.
const pages = [
  ...readdirSync('.').filter((f) => f.endsWith('.html')).sort(),
  ...(existsSync('licensing') ? readdirSync('licensing').filter((f) => f.endsWith('.html')).sort().map((f) => `licensing/${f}`) : []),
];
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const built = [];
/**
 * Refuse to emit a page whose block structure does not close cleanly.
 *
 * A browser silently ignores an unmatched `</div>` — no error, no console
 * warning — so a stray one closes the page wrapper early and everything after it
 * escapes the layout. That shipped to production on the index page and stayed up
 * through several deploys, because nothing in the pipeline was looking.
 */
function assertBalanced(name, body) {
  const TAGS = /<(\/?)(div|section|header|footer|nav|main|article|aside|ul|ol|li|table|figure)\b([^>]*)>/g;
  const stack = [];
  for (const [, close, tag, attrs] of body.matchAll(TAGS)) {
    if (attrs.trimEnd().endsWith('/')) continue;
    if (!close) { stack.push([tag, (attrs.match(/class="([^"]*)"/) || [, ''])[1]]); continue; }
    if (!stack.length) throw new Error(`${name}: unmatched </${tag}> — it would close the page wrapper early`);
    const [open, cls] = stack.pop();
    if (open !== tag) throw new Error(`${name}: </${tag}> closes <${open} class="${cls}">`);
  }
  if (stack.length) throw new Error(`${name}: never closed — ${stack.map(([t, c]) => `<${t} class="${c}">`).join(', ')}`);
}

for (const file of pages) {
  const src = readFileSync(file, 'utf8');
  const body = pageBody(src);
  if (body === null) { console.log(`  skip   ${file} (no metadata comment)`); continue; }
  const slug = file.replace(/\.html$/, '');
  // The scope class is the basename: `pg-licensing/publisher-tos` is not a class.
  const scope = basename(slug);
  const { title, description } = frontmatter(src);
  assertBalanced(`${slug}.html`, body);

  // Every page is titled for the category it documents, not for the site it
  // sits on. A reader arriving from search should learn what this is before
  // they learn whose it is.
  const pageTitle = `Vertical Inference — ${title}`;
  const url = canonical(slug);
  // Escaped for a <script> context: a literal `</script>` inside any string
  // would close the block and leak the remainder as markup.
  const jsonld = JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: title,
    description,
    url,
    isPartOf: { '@type': 'WebSite', name: 'Lloyal HDK documentation', url: `${SITE}/` },
    about: { '@type': 'DefinedTerm', name: 'Vertical Inference', url: CATEGORY_URL },
    publisher: { '@type': 'Organization', name: 'Lloyal Labs', url: 'https://lloyal.ai/' },
  }).replace(/</g, '\\u003c');

  mkdirSync(join(OUT, dirname(slug)), { recursive: true });
  writeFileSync(join(OUT, `${slug}.html`), `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(pageTitle)}</title>
<meta name="description" content="${esc(description)}">
<link rel="canonical" href="${url}">
<meta property="og:type" content="${slug === 'index' ? 'website' : 'article'}">
<meta property="og:site_name" content="Lloyal Labs">
<meta property="og:title" content="${esc(pageTitle)}">
<meta property="og:description" content="${esc(description)}">
<meta property="og:url" content="${url}">
<meta property="og:image" content="${OG_IMAGE}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="${esc(pageTitle)}">
<meta name="twitter:description" content="${esc(description)}">
<meta name="twitter:image" content="${OG_IMAGE}">
<script type="application/ld+json">${jsonld}</script>
<link rel="icon" href="/favicon.svg">
<link rel="stylesheet" href="${CSS}">
</head>
<body id="lloyal-guides" class="${bodyClass(scope)}">
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
// Copied verbatim, not generated. These came out of `docs.json`, the last
// Mintlify artefact in the repo - a 300-line config whose theme, navbar,
// navigation and footer keys nothing has read since the site became static
// files. The redirects were the only live thing in it, and Cloudflare's
// `_redirects` is already their native format.
const redirectSrc = readFileSync('_redirects', 'utf8');
const redirects = redirectSrc.split('\n').filter(Boolean);
writeFileSync(join(OUT, '_redirects'), redirectSrc);

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
const notFound = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not found — Lloyal Labs</title>
<meta name="robots" content="noindex">
<link rel="icon" href="/favicon.svg">
<link rel="stylesheet" href="${CSS}">
</head>
<body id="lloyal-guides" class="${bodyClass('index')}">
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
${notFoundGrid()}
</div>
</section>
<footer class="footer wrap"><span class="mission">Enable every organisation to deliver intelligence on its terms.</span><span>Lloyal Labs &middot; Melbourne</span></footer>
</div>
${NAV_SCRIPT}
</body>
</html>
`;
assertBalanced('404.html', notFound);
writeFileSync(join(OUT, '404.html'), notFound);

// Mintlify generated this; emit an equivalent so it does not vanish silently.
writeFileSync(join(OUT, 'llms.txt'), `# Lloyal HDK documentation

${built.map((p) => `- [${p.title}](https://docs.lloyal.ai/${p.slug === 'index' ? '' : p.slug}): ${p.description}`).join('\n')}
`);

// A sitemap the site has never had. Built from the same `built` list as
// llms.txt, so a page cannot appear in one and be missing from the other.
writeFileSync(join(OUT, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${built.map((p) => `  <url><loc>${canonical(p.slug)}</loc></url>`).join('\n')}
</urlset>
`);

// Cloudflare appends its own managed block to whatever robots.txt is served,
// which is why the site has one today despite the repo never shipping one.
// This adds the part only we can know: where the sitemap is.
writeFileSync(join(OUT, 'robots.txt'), `User-agent: *
Allow: /

Sitemap: ${SITE}/sitemap.xml
`);

console.log(`\n  ${built.length} pages · ${redirects.length} redirects · ${CSS_NAME} → ${OUT}/`);
