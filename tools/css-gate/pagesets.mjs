#!/usr/bin/env node
/**
 * Phase 3 analysis — what scope does each rule group actually need?
 *
 * Groups the sheet by (media, selector-suffix) and reports the set of pages
 * each group is defined on. Scope is DERIVED from that observed pageset and is
 * never hand-assigned, which is what makes leaking structurally impossible: a
 * rule present on four pages gets a four-page prefix, and no step ever widens
 * one.
 *
 *   node tools/css-gate/pagesets.mjs <css> [--verbose]
 */
import { readFileSync } from 'node:fs';
import postcss from 'postcss';
import selectorParser from 'postcss-selector-parser';

const SLUGS = [
  'abilities', 'agent-policy-and-context-pressure', 'build-your-first-harness',
  'continuous-context', 'focal-lens', 'index', 'lookup', 'thinking-in-lloyal',
  'where-a-harness-runs',
];
const LONG = ['agent-policy-and-context-pressure', 'build-your-first-harness',
  'continuous-context', 'thinking-in-lloyal'];
const SHORT = ['abilities', 'focal-lens', 'lookup', 'where-a-harness-runs'];
const HL = [...LONG, 'focal-lens'].sort();

const key = (a) => [...a].sort().join('+');
const NAMED = new Map([
  [key(SLUGS), '.pg'],
  [key(SLUGS.filter((s) => s !== 'index')), '.guide'],
  [key(LONG), '.long'],
  [key(SHORT), '.short'],
  [key(HL), '.hl'],
]);

const css = readFileSync(process.argv[2], 'utf8');
const root = postcss.parse(css);

/** Split "#lloyal-guides.pg-x .foo" into its page and the suffix after the gate. */
function split(sel) {
  let page = null, suffix = sel;
  const t = selectorParser((r) => {
    const first = r.first;
    if (!first) return;
    for (const n of first.nodes) {
      if (n.type === 'combinator') break;
      if (n.type === 'class' && n.value.startsWith('pg-')) page = n.value.slice(3);
    }
    if (page) { first.nodes = first.nodes.filter((n) => !(n.type === 'class' && n.value.startsWith('pg-'))); }
  }).processSync(sel, { lossless: false });
  if (page) suffix = t;
  return { page, suffix };
}

const groups = new Map();
root.walkRules((rule) => {
  const chain = [];
  for (let p = rule.parent; p && p.type === 'atrule'; p = p.parent) chain.unshift(`@${p.name} ${p.params.replace(/\s+/g, '')}`);
  const media = chain.join('&&');
  const { page, suffix } = split(rule.selector);
  if (!page) return;                       // unscoped chrome — band 2, left alone
  const decls = rule.nodes.filter((n) => n.type === 'decl')
    .map((d) => `${d.prop.trim()}:${d.value.trim()}${d.important ? '!important' : ''}`).join(';');
  const k = `${media}||${suffix}`;
  if (!groups.has(k)) groups.set(k, { media, suffix, pages: new Map() });
  const g = groups.get(k);
  if (!g.pages.has(page)) g.pages.set(page, []);
  g.pages.get(page).push(decls);           // multiple = this page defines it more than once
});

const byPageset = new Map();
let multiDefined = 0, identical = 0, differing = 0, single = 0;
for (const g of groups.values()) {
  const pk = key(g.pages.keys());
  const variants = new Set([...g.pages.values()].map((v) => v.join(' ~~ ')));
  if ([...g.pages.values()].some((v) => v.length > 1)) multiDefined++;
  if (g.pages.size === 1) single++;
  else if (variants.size === 1) identical++;
  else differing++;
  if (!byPageset.has(pk)) byPageset.set(pk, { n: 0, size: g.pages.size, pages: [...g.pages.keys()] });
  byPageset.get(pk).n++;
}

console.log(`  groups: ${groups.size}   single-page: ${single}   multi identical: ${identical}   multi differing: ${differing}`);
console.log(`  groups where some page defines the selector more than once: ${multiDefined}`);
console.log();
console.log(`  ${'groups'.padStart(6)}  ${'pages'.padStart(5)}  prefix        pageset`);
for (const [pk, v] of [...byPageset.entries()].sort((a, b) => b[1].n - a[1].n)) {
  const named = NAMED.get(pk) ?? (v.size === 1 ? `.pg-${v.pages[0]}` : '(enumerate)');
  console.log(`  ${String(v.n).padStart(6)}  ${String(v.size).padStart(5)}  ${named.padEnd(13)} ${v.size > 3 && named !== '(enumerate)' ? '' : v.pages.join(', ').slice(0, 70)}`);
}
const covered = [...byPageset.entries()]
  .filter(([pk, v]) => NAMED.has(pk) || v.size === 1)
  .reduce((a, [, v]) => a + v.n, 0);
console.log();
console.log(`  covered by 5 markers + single-page scopes: ${covered}/${groups.size} groups (${Math.round(covered / groups.size * 100)}%)`);
console.log(`  needing an enumerated list:                ${groups.size - covered}`);
