#!/usr/bin/env node
/**
 * Gate D — full-page pixel capture. The human-legible backstop behind Gate A.
 *
 * Captures every document at both sides of every breakpoint in the sheet, at
 * full page height, with rendering forced deterministic. Two runs of the same
 * input must produce byte-identical PNGs, or a later diff means nothing.
 *
 * WHY THE FLAGS: subpixel antialiasing, device scale and font hinting all vary
 * with machine state and would manufacture diffs. Scrollbars change layout
 * width. Web fonts arriving late change metrics mid-capture, so we wait on
 * document.fonts.ready rather than a timeout.
 *
 *   node tools/css-gate/capture.mjs <outdir> [--port 8794]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import puppeteer from 'puppeteer';

const DOCS = [
  'index', 'abilities', 'agent-policy-and-context-pressure', 'build-your-first-harness',
  'continuous-context', 'focal-lens', 'lookup', 'thinking-in-lloyal',
  'where-a-harness-runs', '404',
];

// Both sides of 560, 640, 680, 780, 900/901, 1180 — plus the extremes.
const WIDTHS = [320, 375, 559, 561, 639, 641, 679, 681, 779, 781, 899, 902, 1179, 1181, 1440];

const outdir = process.argv[2];
const port = (process.argv.includes('--port') ? process.argv[process.argv.indexOf('--port') + 1] : '8794');
if (!outdir) { console.error('usage: capture.mjs <outdir> [--port N]'); process.exit(2); }
mkdirSync(outdir, { recursive: true });

const browser = await puppeteer.launch({
  headless: 'shell',
  args: [
    '--force-device-scale-factor=1',
    '--font-render-hinting=none',
    '--disable-lcd-text',
    '--hide-scrollbars',
    '--disable-gpu',
    '--force-color-profile=srgb',
    '--disable-partial-raster',
    '--disable-skia-runtime-opts',
  ],
});

let n = 0;
for (const doc of DOCS) {
  const page = await browser.newPage();
  // Kill animation and transition so nothing is mid-flight at capture time.
  await page.evaluateOnNewDocument(() => {
    const s = document.createElement('style');
    s.textContent = `*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}`;
    document.documentElement.appendChild(s);
  });
  for (const w of WIDTHS) {
    await page.setViewport({ width: w, height: 900, deviceScaleFactor: 1 });
    await page.goto(`http://localhost:${port}/${doc}.html`, { waitUntil: 'networkidle0' });
    await page.evaluate(() => document.fonts.ready);
    const buf = await page.screenshot({ fullPage: true, type: 'png' });
    writeFileSync(join(outdir, `${doc}_${w}.png`), buf);
    n++;
  }
  // print media is a real surface here — nine @print blocks exist
  await page.emulateMediaType('print');
  await page.setViewport({ width: 1180, height: 900, deviceScaleFactor: 1 });
  await page.goto(`http://localhost:${port}/${doc}.html`, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  writeFileSync(join(outdir, `${doc}_print.png`), await page.screenshot({ fullPage: true, type: 'png' }));
  n++;
  await page.close();
}

await browser.close();
console.log(`  captured ${n} renders -> ${outdir}`);
