#!/usr/bin/env node
/**
 * Phase 1 — normalisation. Formatting only.
 *
 * Explodes comma-separated selector lists into one rule each and re-emits the
 * sheet through a deterministic stringifier. Nothing is reordered, deduped or
 * rescoped. `A, B { d }` is exactly equivalent to `A { d } B { d }` at the same
 * position — each selector already carried its own specificity — so Gate A
 * passes by construction.
 *
 * The point is not tidiness. The file arrived as a machine-concatenated export
 * dump with 9,000-character lines and comments sitting *inside* selectors; the
 * later phases need something a script can transform without a regex ever
 * touching a selector.
 *
 *   node tools/css-gate/normalise.mjs <in.css> <out.css>
 */
import { readFileSync, writeFileSync } from 'node:fs';
import postcss from 'postcss';

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) { console.error('usage: normalise.mjs <in.css> <out.css>'); process.exit(2); }

const src = readFileSync(inPath, 'utf8');
const root = postcss.parse(src);

let exploded = 0;
root.walkRules((rule) => {
  const sels = rule.selectors;
  if (sels.length < 2) { rule.selector = sels[0]; return; }
  for (const sel of sels) {
    const clone = rule.clone();
    clone.selector = sel;
    rule.parent.insertBefore(rule, clone);
    exploded++;
  }
  rule.remove();
});

// Deterministic layout: one declaration per line, one selector per rule.
root.walkRules((r) => {
  r.raws.before = r.parent.type === 'root' ? '\n' : '\n  ';
  r.raws.between = ' ';
  r.raws.after = r.parent.type === 'root' ? '\n' : '\n  ';
  r.nodes.forEach((d) => {
    d.raws.before = r.parent.type === 'root' ? '\n  ' : '\n    ';
    if (d.type === 'decl') d.raws.between = ': ';
  });
});
root.walkAtRules((a) => {
  if (a.nodes) { a.raws.before = '\n\n'; a.raws.between = ' '; a.raws.after = '\n'; }
});

const out = root.toString();
writeFileSync(outPath, out);

const lines = (s) => s.split('\n').length;
console.log(`  selectors exploded : ${exploded}`);
console.log(`  lines              : ${lines(src)} -> ${lines(out)}`);
console.log(`  bytes              : ${src.length} -> ${out.length}`);
console.log(`  longest line       : ${Math.max(...src.split('\n').map((l) => l.length))} -> ${Math.max(...out.split('\n').map((l) => l.length))}`);
const firstRule = out.search(/^[^@\s\/]/m);
const importAt = out.indexOf('@import');
console.log(`  @import still first: ${importAt >= 0 && (firstRule < 0 || importAt < firstRule)}`);
