/**
 * Which scopes each page wears. The single source of truth for both the build
 * and the CSS merge tool.
 *
 * WHY THIS FILE EXISTS: these lists used to be written twice — the markers in
 * `tools/css-gate/merge.mjs` deciding what the stylesheet is keyed on, `COHORT`
 * deciding what the `<body>` carries. Adding `wide` to one and not the other
 * silently deleted the root variables block from all nine pages: no `--max`, so
 * every page rendered full-bleed. Gate C caught it, but only after a full build
 * and a minute of rendering. Deriving one from the other makes the mistake
 * unavailable rather than merely detectable.
 *
 * WHY A STATIC TABLE, NOT DERIVED FROM CONTENT: `hl` (the Pygments palette)
 * looks derivable from "does this page contain highlighted spans", and is not —
 * index, lookup, abilities, where-a-harness-runs and 404 all carry such spans
 * that are deliberately UNSTYLED. Adding a page should force a decision here,
 * which is the point; an undeclared slug throws.
 *
 *   guide  every page except index
 *   long   the four long guides    (--max 1180px, looser heading rhythm)
 *   short  the four short ones     (--max 1320px, tighter)
 *   wide   the 1320px measure: the short guides AND index, which is neither
 *          `short` nor `guide` but is where the root variables block lives
 *   hl     the syntax palette: the long guides PLUS focal-lens, not a cohort
 */
export const COHORT = {
  'index':                             ['wide'],
  'build-your-first-harness':          ['guide', 'long', 'hl'],
  'thinking-in-lloyal':                ['guide', 'long', 'hl'],
  'continuous-context':                ['guide', 'long', 'hl'],
  'agent-policy-and-context-pressure': ['guide', 'long', 'hl'],
  'abilities':                         ['guide', 'short', 'wide'],
  'focal-lens':                        ['guide', 'short', 'wide', 'hl'],
  'lookup':                            ['guide', 'short', 'wide'],
  'where-a-harness-runs':              ['guide', 'short', 'wide'],
  // The first page styled entirely by the shared layers — it has no per-page
  // selectors at all, which before the flatten meant no styling at all.
  'publisher-tos':                     ['guide', 'short', 'wide'],
};

/** Every scope a page's <body> carries. Used by the page loop AND by 404.html. */
export const bodyClass = (slug) => {
  // Fail the build rather than ship a page silently missing its cohort — an
  // unstyled page is exactly the failure this whole change exists to remove.
  if (!COHORT[slug]) throw new Error(`no cohort declared for '${slug}' — add it to COHORT in cohorts.mjs`);
  return ['pg', ...COHORT[slug], `pg-${slug}`].join(' ');
};

/**
 * Marker → the exact set of pages wearing it, inverted from the table above.
 *
 * The merge tool will only collapse a group whose page set equals one of these
 * EXACTLY. A page added here but absent from the stylesheet simply stops that
 * marker merging — merges are lost, never correctness.
 */
export const MARKERS = (() => {
  const out = { pg: Object.keys(COHORT) };
  for (const [slug, marks] of Object.entries(COHORT))
    for (const m of marks) (out[m] ??= []).push(slug);
  return out;
})();
