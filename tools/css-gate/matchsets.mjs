#!/usr/bin/env node
/**
 * Which elements does each selector actually match, across the whole corpus?
 *
 * The merge safety check has to answer "can these two rules ever style the same
 * element". Statically that is nearly unanswerable — `.toc a:hover` and
 * `.code-frame code .c` are different specificities of nothing in common, yet
 * `<a class="c">` inside both ancestors is legal, so a sound static analysis has
 * to say "maybe" and block the merge. It blocked all 79 palette rules.
 *
 * But this stylesheet serves exactly ten documents, and they are all built. So
 * the question is not "could any DOM do this" but "does any of OUR ten". That is
 * decidable, exactly, with querySelectorAll — and it is the same standard Gate C
 * already holds the result to.
 *
 * State pseudo-classes are stripped before matching: `querySelectorAll('a:hover')`
 * returns nothing unless a pointer is actually over the element, which would
 * report an empty set and wrongly license a merge. Dropping `:hover` widens the
 * match set, which can only block merges, never permit one.
 *
 *   node tools/css-gate/matchsets.mjs <port> <out.json>
 */
import { readFileSync, writeFileSync } from 'node:fs';
import postcss from 'postcss';
import puppeteer from 'puppeteer';

const DOCS = ['index', 'abilities', 'agent-policy-and-context-pressure',
  'build-your-first-harness', 'continuous-context', 'focal-lens', 'lookup',
  'thinking-in-lloyal', 'where-a-harness-runs', '404'];

const [port, outPath] = process.argv.slice(2);
if (!port || !outPath) { console.error('usage: matchsets.mjs <port> <out.json>'); process.exit(2); }

const COMMENT = /\/\*[\s\S]*?\*\//g;
const STATE = /:(hover|focus|focus-visible|focus-within|active|visited|target|checked|disabled|enabled)\b/g;
const PSEUDO_EL = /::[a-z-]+(\([^)]*\))?/g;

const root = postcss.parse(readFileSync('assets/guides/guides-theme.css', 'utf8'));
const selectors = [];
root.walkRules((r) => {
  for (const s of r.selectors) selectors.push(s.replace(COMMENT, ' ').replace(/\s+/g, ' ').trim());
});
const uniq = [...new Set(selectors)];

/** What we hand the browser: no pseudo-elements, no state. Widening is safe. */
const probe = uniq.map((s) => s.replace(PSEUDO_EL, '').replace(STATE, '').replace(/\s+/g, ' ').trim() || '*');

const browser = await puppeteer.launch({ headless: 'shell', args: ['--hide-scrollbars'] });
const sets = uniq.map(() => []);
let unmatched = 0;

for (let d = 0; d < DOCS.length; d++) {
  const page = await browser.newPage();
  await page.goto(`http://localhost:${port}/${DOCS[d]}.html`, { waitUntil: 'load' });
  const hits = await page.evaluate((sels) => {
    const idx = new Map();
    let n = 0;
    (function walk(el) { idx.set(el, n++); for (const c of el.children) walk(c); })(document.documentElement);
    return sels.map((s) => {
      let els;
      try { els = document.querySelectorAll(s); } catch { return null; }
      return [...els].map((e) => idx.get(e));
    });
  }, probe);
  for (let i = 0; i < uniq.length; i++) {
    if (hits[i] === null) { if (d === 0) unmatched++; sets[i] = null; continue; }
    if (sets[i] !== null) for (const e of hits[i]) sets[i].push(d * 1e6 + e);
  }
  await page.close();
  process.stdout.write(`  ${DOCS[d].padEnd(36)} done\n`);
}
await browser.close();

const out = {};
for (let i = 0; i < uniq.length; i++) out[uniq[i]] = sets[i] === null ? null : [...new Set(sets[i])].sort((a, b) => a - b);
writeFileSync(outPath, JSON.stringify(out));
const empty = Object.values(out).filter((v) => v && !v.length).length;
console.log();
console.log(`  ${uniq.length} distinct selectors · ${empty} match nothing on any page · ${unmatched} unparseable (treated as matching everything)`);
