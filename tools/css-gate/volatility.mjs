#!/usr/bin/env node
/**
 * Phase 0 — find which computed properties are non-deterministic.
 *
 * Gate C compares getComputedStyle for every element before and after a change.
 * That only means something if a no-op comparison is empty, so this loads the
 * same page twice and reports every property that differs. Anything listed here
 * must be excluded from Gate C, and anything NOT listed is fair game.
 *
 * A gate with an unexamined exclusion list is a gate with a hole in it, so the
 * list is derived here rather than guessed.
 *
 *   node tools/css-gate/volatility.mjs [--port 8794]
 */
import puppeteer from 'puppeteer';

const DOCS = ['index', 'continuous-context', 'build-your-first-harness', 'focal-lens', '404'];
const port = process.argv.includes('--port') ? process.argv[process.argv.indexOf('--port') + 1] : '8794';

const dump = async (page, url) => {
  await page.goto(url, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  return page.evaluate(() => {
    const out = [];
    const walk = (el, path) => {
      const cs = getComputedStyle(el);
      const rec = {};
      for (let i = 0; i < cs.length; i++) rec[cs[i]] = cs.getPropertyValue(cs[i]);
      out.push([path, rec]);
      [...el.children].forEach((c, i) => walk(c, `${path}/${i}`));
    };
    walk(document.body, '');
    return out;
  });
};

const browser = await puppeteer.launch({
  headless: 'shell',
  args: ['--force-device-scale-factor=1', '--font-render-hinting=none', '--disable-lcd-text',
    '--hide-scrollbars', '--disable-gpu', '--force-color-profile=srgb'],
});

const volatile = new Map();
let elements = 0, comparisons = 0;

for (const doc of DOCS) {
  const url = `http://localhost:${port}/${doc}.html`;
  const p1 = await browser.newPage();
  await p1.setViewport({ width: 1280, height: 900 });
  const a = await dump(p1, url);
  await p1.close();

  const p2 = await browser.newPage();
  await p2.setViewport({ width: 1280, height: 900 });
  const b = await dump(p2, url);
  await p2.close();

  if (a.length !== b.length) { console.log(`  ${doc}: element count differs (${a.length} vs ${b.length})`); continue; }
  elements += a.length;
  for (let i = 0; i < a.length; i++) {
    for (const k of Object.keys(a[i][1])) {
      comparisons++;
      if (a[i][1][k] !== b[i][1][k]) {
        if (!volatile.has(k)) volatile.set(k, { n: 0, sample: [a[i][1][k], b[i][1][k], doc, a[i][0]] });
        volatile.get(k).n++;
      }
    }
  }
}

await browser.close();

console.log(`  documents: ${DOCS.length}   elements: ${elements}   property comparisons: ${comparisons}`);
console.log();
if (!volatile.size) {
  console.log('  VOLATILITY: none — every computed property is stable across runs.');
  console.log('  Gate C needs no exclusion list.');
} else {
  console.log(`  VOLATILE PROPERTIES: ${volatile.size}`);
  for (const [k, v] of [...volatile.entries()].sort((a, b) => b[1].n - a[1].n)) {
    console.log(`   ${k.padEnd(30)} ${String(v.n).padStart(6)} differences`);
    console.log(`     e.g. ${v.sample[2]} ${v.sample[3] || '<body>'}: ${String(v.sample[0]).slice(0, 50)}  vs  ${String(v.sample[1]).slice(0, 50)}`);
  }
}
