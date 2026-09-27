# Interface design system

SonicSentinel is an operations console. People use it to check what was heard, act on
alerts and decide review cases, so the interface stays quiet. Colour is kept for things
that mean something: severity, audio quality, agreement between the models, and each
model's identity. Everything else is neutral.

## Themes

Light is the default, with a dark theme on the sun/moon button in the top bar. The choice
is saved in the `sst_theme_v2` cookie (`light` or `dark`) and applied on the server, so the
page doesn't flash on load. It is a browser setting, not an account setting.

All values are in `static/css/tokens.css`:

| Token group | Light | Dark | Used for |
|---|---|---|---|
| Surfaces | `#f4f7fe` canvas, `#ffffff` panels | `#0a0e16` canvas, `#111723` panels | page, panels, inputs |
| Text | `#1b2559` / `#4a5578` / `#68759a` | `#e8ecf3` / `#b0bacb` / `#8a97ad` | primary, secondary, tertiary |
| Accent | `#4361ee` | `#6f9dff` | primary button, links, focus, active nav |
| Severity | grey, blue, orange, red ramp | lighter versions of the same hues | alert and event badges |
| Models | Python `#0891b2`, TM `#c026d3` | Python `#22d3ee`, TM `#e879f9` | model cards, chips, bars |

Type is Geist and Geist Mono, served from `static/fonts/`.

Severity, quality and agreement colours never appear on their own: every badge also has
its text label.

## Layout and components

- On desktop there is a white navigation rail with a blue pill on the active item. Below
  1025px it becomes a drawer opened from the top bar, and below 721px tables turn into
  stacked cards.
- The dashboard opens with a greeting, the upload and live-monitor buttons, and event
  totals. The latest detection, charts and timeline below it all come from the database.
- Cards, tiles, tables and forms share one surface style: white, rounded, a soft shadow,
  no gradients. Numbers use tabular figures.
- One filled primary button per area; the rest are outline or ghost buttons.
- The console shares its buttons, cards and colours with the public page at `/`.

`static/css/app.css` has the base components, `product.css` the page components, and
`design.css` the current look (loaded last). `landing.css` is only for the public page. All
of them take colours, sizes and radii from the tokens.

## Motion

Pages fade in on load, sections reveal on scroll on the public page, numbers count up, the
dashboard chart draws itself, and cards lift slightly on hover. The pulsing dot while the
microphone is live and the upload spinner are the only motion that carries meaning. The
operating system's reduced-motion setting turns all of it off, and every animated element
ends in its final state, so nothing is hidden.

## Accessibility

- Visible focus ring on every interactive element (`--focus-ring`).
- Body text contrast of at least 4.5:1 in both themes.
- Status is never shown by colour alone.
- Live regions announce upload progress, live-window results and toasts.
- Forced-colours mode uses the operating system's palette.

## Checking it

`.venv/bin/python tools/check_ui.py` (needs Playwright and Google Chrome) runs the app
against a temporary database with the models off. It checks the theme toggle, signs in
through the real form, opens every page at 320, 390, 768 and 1440 px looking for horizontal
overflow, broken images and unlabelled links, and saves screenshots to `screenshots/ui/`
and a report to `reports/ui_review.json`. It is not a full accessibility audit; Safari,
Firefox and a real microphone still need a manual check.
