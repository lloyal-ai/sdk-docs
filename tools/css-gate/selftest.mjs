#!/usr/bin/env node
/**
 * Gate A self-test. A gate is only worth its failures.
 *
 * Mutates the real stylesheet four ways — one per class of regression this
 * refactor can produce — and asserts the gate rejects each. Identity must pass.
 *
 * WHY EACH MUTATION ASSERTS ITS OWN INPUT CHANGED: the first draft of the
 * specificity test silently patched a string that did not exist (the selector
 * was comma-grouped), so the gate "passed" a mutation that never happened. A
 * green result on an unapplied mutation is worse than no test.
 */
import { readFileSync } from 'node:fs';
import postcss from 'postcss';
import { compare } from './gate-a.mjs';

const CSS = 'assets/guides/guides-theme.css';
const base = readFileSync(CSS, 'utf8');
const bodyClassFor = (slug) => `pg-${slug}`;

/** Drop a scope class — specificity (1,2,0) -> (1,1,0). The plan's original bug. */
function mutSpecificity(css) {
  const root = postcss.parse(css);
  let hit = null;
  root.walkRules((r) => {
    if (hit) return;
    const sels = r.selectors;
    const i = sels.findIndex((s) => /^#lloyal-guides\.pg-[a-z-]+\s+\.\w/.test(s));
    if (i < 0) return;
    hit = sels[i];
    sels[i] = sels[i].replace(/#lloyal-guides\.pg-[a-z-]+/, '#lloyal-guides');
    r.selectors = sels;
  });
  return { css: root.toString(), note: hit };
}

/** Widen a single-cohort rule so it starts matching pages that never had it. */
function mutLeak(css) {
  const root = postcss.parse(css);
  let hit = null;
  root.walkRules((r) => {
    if (hit) return;
    const sels = r.selectors;
    const i = sels.findIndex((s) => /^#lloyal-guides\.pg-thinking-in-lloyal\s/.test(s));
    if (i < 0) return;
    hit = sels[i];
    sels[i] = sels[i].replace('#lloyal-guides.pg-thinking-in-lloyal', '#lloyal-guides');
    r.selectors = sels;
  });
  return { css: root.toString(), note: hit };
}

/** Change one declaration value. */
function mutValue(css) {
  const i = css.indexOf('--max:1320px');
  if (i < 0) return { css, note: null };
  return { css: css.slice(0, i) + '--max:1321px' + css.slice(i + 12), note: '--max 1320px -> 1321px' };
}

/** Swap two adjacent rules — same content, different cascade order. */
function mutOrder(css) {
  const root = postcss.parse(css);
  let a = null, b = null;
  root.walkRules((r) => {
    if (b) return;
    if (!/^#lloyal-guides\.pg-index\s+\./.test(r.selector)) return;
    if (!a) a = r; else b = r;
  });
  if (!a || !b) return { css, note: null };
  const sa = a.selector, sb = b.selector;
  const da = a.nodes.map((n) => n.toString()).join(';');
  a.selector = sb; b.selector = sa;
  a.removeAll(); postcss.parse(`x{${b.nodes.map((n) => n.toString()).join(';')}}`).first.nodes.forEach((n) => a.append(n.clone()));
  b.removeAll(); postcss.parse(`x{${da}}`).first.nodes.forEach((n) => b.append(n.clone()));
  return { css: root.toString(), note: `${sa} <-> ${sb}` };
}

const cases = [
  ['identity  (must PASS)', (c) => ({ css: c, note: 'unchanged' }), false],
  ['specificity drop', mutSpecificity, true],
  ['silent leak', mutLeak, true],
  ['value change', mutValue, true],
  ['rule reorder', mutOrder, true],
];

let bad = 0;
for (const [name, mutate, expectFail] of cases) {
  const { css, note } = mutate(base);
  const applied = css !== base;
  if (expectFail && !applied) {
    console.log(`  ${name.padEnd(24)} INVALID — mutation was a no-op, proves nothing`);
    bad++;
    continue;
  }
  const failed = compare(base, css, bodyClassFor).length > 0;
  const ok = failed === expectFail;
  if (!ok) bad++;
  console.log(`  ${name.padEnd(24)} ${ok ? 'ok  ' : 'BAD '} gate ${failed ? 'rejected' : 'accepted'}${note ? `  (${String(note).slice(0, 58)})` : ''}`);
}

console.log();
console.log(bad ? `  SELFTEST: ${bad} case(s) wrong` : '  SELFTEST: all cases behaved as required');
process.exit(bad ? 1 : 0);
