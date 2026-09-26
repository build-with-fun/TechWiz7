# Interface design system

SonicSentinel is an operations console. People use it to check what was heard, act on
alerts and decide review cases, so the interface is quiet on purpose. Colour is kept for
things that mean something: severity, audio quality, agreement between the models, and the
two models' identities. Everything else is neutral.

## Themes

Two themes, dark (default) and light, switched with the sun/moon button in the header. The
choice is stored in the `sst_theme` cookie (`dark` or `light`; anything else falls back to
dark), and the server renders it, so pages do not flash on load. It is a browser preference,
not an account setting.

All values live in `static/css/tokens.css`:

| Token group | Dark | Light | Used for |
|---|---|---|---|
| Surfaces | `#0f1216` canvas, `#161a20` panels | `#f4f5f7` canvas, `#ffffff` panels | page, panels, inputs |
| Text | `#e7eaef` / `#aab2be` / `#8e97a5` | `#151a21` / `#465160` / `#5a6573` | primary, secondary, tertiary |
| Accent | `#6cb2ff` | `#1f68c9` | primary button, links, focus, active nav |
| Severity | informational → critical ramp | same hues, darker text | alert and event badges |
| Models | Python `#38bdf8`, TM `#c084fc` | darker variants | model cards, chips, bars |

Severity, quality and consistency colours never appear alone: every badge also carries its
text label and a glyph.

## Layout and components

- Sticky top bar and a left navigation rail on desktop; icon-only rail on tablets; a
  horizontal scrolling nav under the top bar on phones.
- The dashboard opens with a status strip: which model versions are in service, their
  measured test accuracy, and the two main actions (upload, live monitor). The metrics,
  charts and timeline below come from the database.
- Panels, metric tiles, tables and forms share one surface style: a 1px border, 8–10px
  radius, no gradients, no glass, no glow. Numbers use tabular figures.
- One primary (filled) button per area; everything else is an outline or ghost button.
- Sign-in and registration are a single centred card.

`static/css/app.css` holds the base components and responsive shell; `static/css/product.css`
holds the page components. Both take every colour, size and radius from the tokens.

## Motion

Motion is feedback only: hover and focus colour changes, bar fills, the pulsing dot while the
microphone is live, the upload spinner. There are no decorative or entrance animations.
The operating system's reduced-motion setting removes all of it.

## Accessibility

- Visible focus ring on every interactive element (`--focus-ring`).
- Text contrast is at least 4.5:1 for body text in both themes.
- Status is never colour-only (text + glyph on every badge).
- Live regions announce upload progress, live-window results and toasts.
- Forced-colours mode hands the palette to the operating system.

## Verification

`.venv/bin/python tools/check_ui.py` (needs Playwright and Google Chrome) runs the real Flask
UI against a temporary seeded database with model loading off. It checks the theme toggle and
its persistence, signs in through the real form, visits every navigation route at 320, 390,
768 and 1440 px for horizontal overflow, broken images and unnamed links, and saves
screenshots to `screenshots/ui/` plus a report in `reports/ui_review.json`. It is not a full
accessibility audit, and Safari, Firefox and a physical microphone still need a manual pass.
