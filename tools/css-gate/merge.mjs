#!/usr/bin/env node
/**
 * Phase 3 — collapse per-page duplicates onto the cohort markers.
 *
 * A group is (media context, selector suffix, declarations). If the set of pages
 * declaring that exact group equals a marker's page set EXACTLY, the group can
 * become one rule prefixed with that marker — but only if it can be placed
 * somewhere that reorders nothing. Everything else is left alone, duplicated as
 * today.
 *
 * WHY EXACT SET EQUALITY, NEVER SUBSET: a rule may only lose its per-page gate
 * if every page behind the new gate already had it. Loosening this by one page
 * is a leak, and 51 groups in this file are built from classes present on every
 * page while being declared on a strict subset — they look global and are not.
 *
 * WHY EVERY PREFIX KEEPS A CLASS: `#lloyal-guides.pg` is (1,1,0), the same as
 * `#lloyal-guides.pg-index`. Dropping to bare `#lloyal-guides` is (1,0,0) and
 * would re-decide 61 order-dependent overrides by specificity instead of source
 * order — invisible in review, because the selector merely looks cleaner.
 *
 * WHY THE SAFETY CHECK EXISTS (this cost a Gate C failure to learn): merging
 * moves a page's copy of a rule from where that page declared it to wherever the
 * merged rule lands. Every equal-specificity rule it jumps over changes places
 * with it, and equal specificity is decided by source order. A first attempt
 * that ignored this shipped `font-size: 18px -> 25px` at 320px on eight pages
 * and lost the entire print stylesheet on four.
 *
 * The Phase 0 survey concluded all 370 groups were safely mergeable. That was
 * wrong, and the flaw is worth naming: it filtered conflicts to pairs sharing an
 * IDENTICAL media context. But `@media print` and an unmediated rule are both
 * live while printing, and a `max-width` rule and an unmediated rule are both
 * live below the breakpoint. Co-activity is the real relation, and textual
 * equality is not even a subset of it. Two rules are treated as unable to
 * interfere only where that is provable: disjoint width intervals, print vs
 * screen, different subject tags, different pseudo-elements. Everything else is
 * assumed to interfere, which costs merges and never correctness.
 *
 * WHY THE MATCH-SET ORACLE IS NOT OPTIONAL: run without it and this file merges
 * 392 rules; run with it, 1,053. The syntax palette is the clearest case — 79
 * rules all setting `color` at one specificity on classes nothing static can
 * prove disjoint, so each reads as a conflict with the other 78 and none moves.
 * Measured against the ten documents, they collide with nothing.
 *
 * WHY SELECTORS ARE CLEANED FIRST: this file writes that palette with a CSS
 * comment sitting INSIDE the selector, between the scope and the descendant part
 * — and off by one against the token it appears to label. Comments are inert to
 * a browser but not to a grouping key, so five identical palettes look like five
 * different rules. Stripping them is sound only where whitespace already
 * separates the parts: a comment with no space on either side joins its
 * neighbours into one compound rather than a descendant pair, so any rule shaped
 * that way is left alone rather than guessed at.
 *
 *   node tools/css-gate/merge.mjs <in> <out> pg,guide[,long,short,hl] [matchsets.json]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import postcss from 'postcss';
import selectorParser from 'postcss-selector-parser';

const LONG = ['agent-policy-and-context-pressure', 'build-your-first-harness',
  'continuous-context', 'thinking-in-lloyal'];
const SHORT = ['abilities', 'focal-lens', 'lookup', 'where-a-harness-runs'];
const ALL = [...LONG, ...SHORT, 'index'];

/** Marker → the page set it stands for. */
const MARKERS = {
  pg:    ALL,
  guide: [...LONG, ...SHORT],
  long:  LONG,
  short: SHORT,
  hl:    [...LONG, 'focal-lens'],   // the palette is long PLUS focal-lens, not a cohort
};

const [inPath, outPath, want, matchPath] = process.argv.slice(2);
if (!inPath || !outPath || !want) { console.error('usage: merge.mjs <in> <out> pg,guide,... [matchsets.json]'); process.exit(2); }
const ACTIVE = want.split(',').map((s) => s.trim()).filter(Boolean);
for (const m of ACTIVE) if (!MARKERS[m]) { console.error(`unknown marker: ${m}`); process.exit(2); }

const key = (arr) => [...new Set(arr)].sort().join('|');
const TARGET = new Map(ACTIVE.map((m) => [key(MARKERS[m]), m]));

const root = postcss.parse(readFileSync(inPath, 'utf8'));

/** Media chain for a rule, as written — never normalised into a comparison. */
const mediaOf = (rule) => {
  const chain = [];
  for (let p = rule.parent; p && p.type === 'atrule'; p = p.parent) chain.unshift(`@${p.name} ${p.params}`);
  return chain.join(' && ');
};

/**
 * A selector with comments removed, or null when that cannot be done safely.
 * A comment with whitespace beside it separates two compounds; one without
 * whitespace joins them. Only the first case is rewritable.
 */
const COMMENT = /\/\*[\s\S]*?\*\//g;
function cleanSel(sel) {
  for (const m of sel.matchAll(COMMENT)) {
    const before = sel[m.index - 1], after = sel[m.index + m[0].length];
    if (before !== undefined && after !== undefined && !/\s/.test(before) && !/\s/.test(after)) return null;
  }
  return sel.replace(COMMENT, ' ').replace(/\s+/g, ' ').trim();
}

/** Split `#lloyal-guides.pg-<slug><rest>` into its page and its suffix. */
const SCOPE = /^#lloyal-guides\.pg-([a-z0-9-]+)/;
const split = (sel) => {
  const m = sel.match(SCOPE);
  return m ? { page: m[1], suffix: sel.slice(m[0].length) } : null;
};

/**
 * Specificity triple, plus the tag and pseudo-element of the SUBJECT compound —
 * the element the rule actually styles. String munging would misread `>` here,
 * so this walks a real parse tree.
 */
function analyse(sel) {
  let a = 0, b = 0, c = 0, cur = { tag: null, pe: null };
  selectorParser((r) => {
    const first = r.first;
    if (first) for (const n of first.nodes) {
      if (n.type === 'combinator') { cur = { tag: null, pe: null }; continue; }
      if (n.type === 'tag') cur.tag = n.value;
      if (n.type === 'pseudo' && n.value.startsWith('::')) cur.pe = n.value;
    }
    r.walk((n) => {
      if (n.type === 'id') a++;
      else if (n.type === 'class' || n.type === 'attribute') b++;
      else if (n.type === 'pseudo') (n.value.startsWith('::') ? c++ : b++);
      else if (n.type === 'tag') c++;
    });
  }).processSync(sel, { lossless: false });
  return { spec: `${a},${b},${c}`, tag: cur.tag, pseudoEl: cur.pe };
}

/**
 * Do two rules ever style the same element? Answered against the corpus, not in
 * the abstract — see matchsets.mjs. Without it the honest static answer is
 * "maybe" for almost every pair (`.toc a:hover` and `.code-frame code .c` share
 * nothing, yet an `<a class="c">` under both ancestors is legal), and "maybe"
 * blocks the merge. With it, the answer is measured on the ten documents this
 * stylesheet actually serves.
 *
 * A missing or unparseable set means "assume it collides". A differing
 * pseudo-element still means no collision — the probe strips those, so two rules
 * on the same element but different boxes would otherwise look like a conflict.
 */
const MS = matchPath ? JSON.parse(readFileSync(matchPath, 'utf8')) : null;
function canCollide(x, y) {
  if (x.pseudoEl !== y.pseudoEl) return false;
  if (!MS) return !(x.tag && y.tag && x.tag !== y.tag);
  const a = MS[x.matchKey], b = MS[y.matchKey];
  if (!a || !b) return true;
  if (!a.length || !b.length) return false;
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) return true;
    a[i] < b[j] ? i++ : j++;
  }
  return false;
}

/**
 * Media chains as a width interval plus an optional type. Anything with a
 * feature this does not model is `opaque` and assumed co-live with everything.
 */
function mediaRange(chain) {
  if (!chain) return { min: 0, max: Infinity, type: null, opaque: false };
  let min = 0, max = Infinity, type = null, opaque = false;
  for (const part of chain.split('&&')) {
    let rest = part.replace(/^\s*@media\s*/, '').trim();
    const t = rest.match(/^(only\s+)?(print|screen)\b/);
    if (t) { type = t[2]; rest = rest.slice(t[0].length).replace(/^\s*and\s*/, ''); }
    for (const cond of rest.split(/\s+and\s+/).filter(Boolean)) {
      const w = cond.match(/^\(\s*(min|max)-width\s*:\s*(\d+(?:\.\d+)?)px\s*\)$/);
      if (!w) { opaque = true; continue; }
      if (w[1] === 'min') min = Math.max(min, +w[2]); else max = Math.min(max, +w[2]);
    }
  }
  return { min, max, type, opaque };
}
const coLive = (x, y) => {
  if (x.opaque || y.opaque) return true;
  if (x.type && y.type && x.type !== y.type) return false;
  return x.min <= y.max && y.min <= x.max;
};

/**
 * Do two property names contend for the same computed value? Name equality is
 * not enough: `background:transparent!important` and `background-color:#272822`
 * are the same fight, and this file has exactly that pair three times over.
 */
const ALIAS = { font: ['line-height', 'font'], inset: ['top', 'right', 'bottom', 'left', 'inset'] };
const expand = (p) => ALIAS[p] ?? [p];
const contends = (x, y) => {
  if (x === 'all' || y === 'all') return true;
  for (const a of expand(x)) for (const b of expand(y))
    if (a === b || a.startsWith(`${b}-`) || b.startsWith(`${a}-`)) return true;
  return false;
};

// ---- index every rule ------------------------------------------------------
const rules = [];
root.walkRules((rule) => {
  const raw = rule.selector.trim();
  const clean = cleanSel(raw);
  const s = rule.selectors.length === 1 && clean !== null ? split(clean) : null;
  const media = mediaOf(rule);
  const { spec, tag, pseudoEl } = analyse(clean ?? raw);
  rules.push({
    node: rule, i: rules.length, media, range: mediaRange(media), spec, tag, pseudoEl,
    matchKey: clean ?? raw,
    // a rule with no page scope applies to every page, so it can block any merge
    page: s && ALL.includes(s.page) ? s.page : null,
    suffix: s ? s.suffix : null,
    props: rule.nodes.filter((n) => n.type === 'decl').map((d) => d.prop.toLowerCase().trim()),
    decls: rule.nodes.map((n) => n.toString().trim()).join(';'),
  });
});
const appliesTo = (r, page) => r.page === null || r.page === page;

// ---- group -----------------------------------------------------------------
const groups = new Map();
for (const r of rules) {
  if (r.page === null || r.suffix === null) continue;
  const k = JSON.stringify([r.media, r.suffix, r.decls]);
  if (!groups.has(k)) groups.set(k, { k, suffix: r.suffix, pages: [], members: [] });
  const g = groups.get(k);
  g.pages.push(r.page);
  g.members.push(r);
  r.group = g;
}
for (const g of groups.values()) {
  // a page listed twice is not a clean set, so it is never a merge candidate
  g.marker = new Set(g.pages).size === g.pages.length ? TARGET.get(key(g.pages)) ?? null : null;
}

/**
 * Can every rule in `unit` move to its paired destination without swapping past
 * an equal-specificity rule that contends for one of its properties? Rules
 * inside the unit are exempt: the unit moves whole, so their order is preserved.
 */
function safeMove(unit, moves) {
  for (const [m, t] of moves) {
    if (m.i === t) continue;
    const lo = Math.min(t, m.i), hi = Math.max(t, m.i);
    for (let j = lo; j <= hi; j++) {
      const x = rules[j];
      if (x === m || unit.has(x)) continue;
      if (x.spec !== m.spec) continue;                    // specificity decides; order cannot
      if (!appliesTo(x, m.page)) continue;                // never co-live on this page
      if (!coLive(x.range, m.range)) continue;            // never live at the same width or media
      if (!canCollide(x, m)) continue;                    // never the same element
      if (x.props.some((p) => m.props.some((q) => contends(p, q)))) return false;
    }
  }
  return true;
}

const hits = new Map(ACTIVE.map((m) => [m, 0]));
const blocked = new Map(ACTIVE.map((m) => [m, 0]));
let removed = 0;
const done = new Set();

/** Rewrite `keep` onto the marker and delete the copies it now stands for. */
function apply(marker, keep, copies) {
  keep.node.selector = `#lloyal-guides.${marker}${keep.group.suffix}`;
  for (const c of copies) { c.node.remove(); removed++; }
  hits.set(marker, hits.get(marker) + 1);
  done.add(keep.group);
}

// ---- merge ----------------------------------------------------------------
for (const g of groups.values()) {
  if (done.has(g) || !g.marker) continue;
  const first = g.members[0], last = g.members[g.members.length - 1];
  const unit = new Set(g.members);
  const keep = safeMove(unit, g.members.map((m) => [m, first.i])) ? first
    : safeMove(unit, g.members.map((m) => [m, last.i])) ? last : null;
  if (!keep) { blocked.set(g.marker, blocked.get(g.marker) + 1); continue; }
  apply(g.marker, keep, g.members.filter((m) => m !== keep));
}

// an emptied @media block would change nothing, but leaving it is noise
root.walkAtRules((at) => { if (at.nodes && at.nodes.length === 0) at.remove(); });

writeFileSync(outPath, root.toString());
for (const m of ACTIVE) {
  console.log(`  .${m.padEnd(6)} ${String(hits.get(m)).padStart(4)} collapsed   ${String(blocked.get(m)).padStart(4)} left alone (would reorder)`);
}
console.log(`  ${removed} duplicate rules removed`);
