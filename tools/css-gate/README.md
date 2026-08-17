# css-gate

Regression gates for `assets/guides/guides-theme.css`.

The stylesheet is scoped per page as `#lloyal-guides.pg-<slug>`, so three
quarters of it is duplication and a new page renders completely unstyled until
~120 selectors are cloned onto its slug. These tools exist to make changing that
arrangement provable rather than hopeful.

## The gates

| script | proves | when |
|---|---|---|
| `gate-a.mjs` | every document cascades the identical **ordered** rule list | order-preserving changes; permanent leak detector |
| `gate-c.mjs` | the browser resolves **every element and property** identically | the arbiter for any change that reorders |
| `capture.mjs` | full-page pixel renders, all breakpoints + print | human-legible backstop |
| `selftest.mjs` | the gates actually **reject** the four regression classes | before trusting any of the above |
| `pagesets.mjs` | which pages each rule group is defined on | deriving scope instead of assigning it |
| `volatility.mjs` | which computed properties are non-deterministic | establishing Gate C's exclusion list (currently: none needed) |

## Two things learned the hard way

**A gate is only worth its failures.** `selftest.mjs` mutates the real
stylesheet four ways — specificity drop, silent leak, changed value, reordered
rules — and requires the gate to reject each. Its first draft "passed" a
mutation that never applied, because the selector it patched was comma-grouped
and the literal string did not exist. Every mutation now asserts its own input
actually changed; a green result on an unapplied mutation is worse than no test.

**Gate A is stricter than correctness.** It demands an identical ordered list,
but the nine pages disagree on the relative order of 14,016 rule pairs. None of
those can change an outcome — filtering to pairs that share a media context,
share a property *and* can match the same element leaves zero — so a merged
sheet is correct while failing Gate A. That is why Gate C exists and why it, not
Gate A, adjudicates the merge.

## Running

```sh
npm run css:selftest                 # do the gates reject what they must?
npm run css:pagesets                 # what scope does each group actually need?
npm run css:volatility               # is computed style deterministic here?

node build.mjs && (cd dist && python3 -m http.server 8794 &)
node tools/css-gate/capture.mjs /tmp/before --port 8794
node tools/css-gate/gate-a.mjs old.css new.css
node tools/css-gate/gate-c.mjs 8794 8795     # two builds, two ports
```

`gate-c.mjs` expects the old and new builds served on separate ports so it can
snapshot both in one browser session — same process, same fonts, same rasteriser.
