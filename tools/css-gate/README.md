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

## Serving the two builds

Both gates compare a running old build against a running new one, so the servers
have to be right or the result is meaningless. Two things go wrong:

1. **They die with the shell.** Start them detached — `(cd dir && nohup python3
   -m http.server PORT >/dev/null 2>&1 &)`.
2. **The port is already taken.** This is the dangerous one. A server left over
   from an earlier session keeps the port, the new bind fails silently, and the
   gate happily compares against whatever that other server is serving. It cost
   a full Gate C run reporting `ELEMENT COUNT: 5 -> 255` — not a regression, just
   a different website.

So always prove the bind before gating:

```sh
curl -s http://localhost:PORT/abilities.html | wc -c   # must equal the file on disk
```

`lsof -nP -iTCP -sTCP:LISTEN | grep :88` shows what is squatting.

## Order of operations for a merge

```sh
node build.mjs && cp -R dist /tmp/_base      # baseline, serve it and VERIFY
node tools/css-gate/matchsets.mjs 8811 /tmp/_ms.json
node tools/css-gate/merge.mjs assets/guides/guides-theme.css /tmp/_m.css pg,guide,long,short,hl /tmp/_ms.json
cp /tmp/_m.css assets/guides/guides-theme.css && node build.mjs
node tools/css-gate/gate-c.mjs 8811 8812     # the arbiter
```

The match sets must be built from the **baseline** stylesheet, before the merge
rewrites any selector — they are the evidence, not a product of the change.
