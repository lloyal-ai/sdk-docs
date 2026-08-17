#!/usr/bin/env node
/**
 * Gate C — computed-style equivalence. The arbiter for any change that reorders.
 *
 * Loads each document from two builds and compares getComputedStyle for every
 * element, every property, at both sides of every breakpoint plus print. An
 * empty diff means the browser resolves every element identically, which is the
 * actual requirement.
 *
 * WHY NOT GATE A FOR THIS: Gate A demands an identical *ordered* rule list, but
 * the nine pages disagree on the relative order of 14,016 rule pairs. None can
 * change an outcome (filtering to pairs sharing a media context, a property and
 * a possible common element leaves zero), so a merged sheet is correct while
 * failing Gate A. Gate A proves order-preserving changes; this measures results.
 *
 * VERIFIED PRECONDITION: a no-op comparison is empty. 855,552 property
 * comparisons over 2,228 elements showed zero volatility, so there is no
 * exclusion list and every difference reported here is real.
 *
 * WHY PSEUDOS ARE QUERIED SEPARATELY: sweeping ::before/::after/::marker on
 * every element cost 5.6x (2,215ms vs 394ms per snapshot) to cover the 13 rules
 * in the sheet that target a pseudo-element. They are now read only on the
 * elements those selectors actually match.
 *
 *   node tools/css-gate/gate-c.mjs <portOld> <portNew> [--quick]
 */
import { readFileSync } from 'node:fs';
import postcss from 'postcss';
import puppeteer from 'puppeteer';

const DOCS = ['index', 'abilities', 'agent-policy-and-context-pressure',
  'build-your-first-harness', 'continuous-context', 'focal-lens', 'lookup',
  'thinking-in-lloyal', 'where-a-harness-runs', '404'];
const FULL = [320, 559, 561, 639, 641, 679, 681, 779, 781, 899, 902, 1179, 1181, 1440];
const QUICK = [375, 561, 899, 1181];

const [portOld, portNew] = process.argv.slice(2);
if (!portOld || !portNew) { console.error('usage: gate-c.mjs <portOld> <portNew> [--quick]'); process.exit(2); }
const WIDTHS = process.argv.includes('--quick') ? QUICK : FULL;

/** Selectors in the sheet that target a pseudo-element, minus the unreadable ones. */
function pseudoSelectors(cssPath) {
  const root = postcss.parse(readFileSync(cssPath, 'utf8'));
  const out = new Set();
  root.walkRules((r) => {
    for (const sel of r.selectors) {
      const m = sel.match(/::[a-z-]+/);
      if (!m) continue;
      if (m[0].startsWith('::-webkit-scrollbar')) continue;   // not readable via getComputedStyle
      out.add(JSON.stringify([sel.slice(0, sel.indexOf(m[0])).trim(), m[0]]));
    }
  });
  return [...out].map((s) => JSON.parse(s));
}
const PSEUDOS = pseudoSelectors('assets/guides/guides-theme.css');

/**
 * Navigation, not style reading, dominates: `networkidle0` costs ~1s per goto
 * and a full matrix needs hundreds. Each document is loaded ONCE per build and
 * then only resized — layout and the cascade recompute on viewport change, so
 * re-navigating buys nothing.
 */
const load = async (page, url) => {
  await page.goto(url, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
};

const snapshot = async (page, pseudos) =>
  page.evaluate((ps) => {
    const rows = [];
    const read = (el, pseudo) => {
      const cs = getComputedStyle(el, pseudo);
      const rec = [];
      for (let i = 0; i < cs.length; i++) rec.push(cs[i] + '' + cs.getPropertyValue(cs[i]));
      return rec.join('');
    };
    const walk = (el, path) => {
      rows.push(path + '' + read(el, null));
      [...el.children].forEach((c, i) => walk(c, path + '/' + i));
    };
    walk(document.body, '');
    // only where a rule actually targets a pseudo-element
    for (const [base, pseudo] of ps) {
      let els = [];
      try { els = [...document.querySelectorAll(base || '*')]; } catch { continue; }
      els.forEach((el, i) => rows.push(`${base}${pseudo}#${i}${read(el, pseudo)}`));
    }
    return rows;
  }, pseudos);

const browser = await puppeteer.launch({
  headless: 'shell',
  args: ['--force-device-scale-factor=1', '--font-render-hinting=none', '--disable-lcd-text',
    '--hide-scrollbars', '--disable-gpu', '--force-color-profile=srgb'],
});

const diffs = [];
let snapshots = 0;
const t0 = Date.now();

for (const doc of DOCS) {
  const pa = await browser.newPage();
  const pb = await browser.newPage();
  await load(pa, `http://localhost:${portOld}/${doc}.html`);
  await load(pb, `http://localhost:${portNew}/${doc}.html`);
  for (const w of [...WIDTHS, 'print']) {
    const width = w === 'print' ? 1180 : w;
    for (const p of [pa, pb]) {
      await p.setViewport({ width, height: 900 });
      await p.emulateMediaType(w === 'print' ? 'print' : null);
    }
    const a = await snapshot(pa, PSEUDOS);
    const b = await snapshot(pb, PSEUDOS);
    snapshots += 2;
    if (a.length !== b.length) { diffs.push({ doc, w, prop: 'ELEMENT COUNT', old: a.length, new: b.length }); continue; }
    for (let i = 0; i < a.length; i++) {
      if (a[i] === b[i]) continue;
      const [path, pa_] = a[i].split('');
      const [, pb_] = b[i].split('');
      const A = new Map(pa_.split('').map((s) => s.split('')));
      const B = new Map(pb_.split('').map((s) => s.split('')));
      for (const [k, v] of A) if (B.get(k) !== v) diffs.push({ doc, w, path, prop: k, old: v, new: B.get(k) });
    }
  }
  await pa.close(); await pb.close();
  console.log(`  ${doc.padEnd(36)} ${diffs.length ? `${diffs.length} diffs` : 'clean'}`);
}

await browser.close();
console.log();
console.log(`  ${snapshots} snapshots · ${PSEUDOS.length} pseudo selectors · ${Math.round((Date.now() - t0) / 1000)}s`);
if (!diffs.length) { console.log('  GATE C: PASS — computed style identical everywhere'); process.exit(0); }
console.log(`  GATE C: FAIL — ${diffs.length} differences`);
const seen = new Set();
for (const d of diffs) {
  const k = `${d.doc}|${d.prop}`;
  if (seen.has(k) || seen.size > 25) continue;
  seen.add(k);
  console.log(`   ${d.doc} @${d.w} ${d.path ?? ''} ${d.prop}: ${String(d.old).slice(0, 44)} -> ${String(d.new).slice(0, 44)}`);
}
process.exit(1);
