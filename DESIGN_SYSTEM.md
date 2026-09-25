# Interface design system

SonicSentinel is an operations console: operators need to notice uncertain classifications and alerts quickly, then inspect the evidence without losing context. The visual language uses deep navy surfaces, blue/cyan signals for navigation and active controls, and restrained amber/red for warnings and critical outcomes. Gradients identify the product and create depth; they are not used as the sole carrier of status.

## Foundations

The shared stylesheet is `static/css/product.css`, loaded after `static/css/app.css`. It gives the existing Jinja pages a coherent vocabulary rather than replacing every template. The login uses a split hero and a CSS sound-wave motif; application pages use a left navigation rail, page title, compact metrics, and evidence panels. On phones, the rail becomes a single-row horizontal navigation scroller that centers the active page. Responsive rules also collapse content grids. Motion is limited to hover/focus and entry transitions; `prefers-reduced-motion` disables nonessential animation.

| Element | Use |
|---|---|
| Deep navy background | Keeps long sessions comfortable and lets evidence panels stand out |
| Blue/cyan gradient | Primary actions, focus, active navigation and sound-wave identity |
| Amber and red | Escalation and critical conditions, paired with text labels |
| Rounded panels and quiet borders | Group an event's scores, quality, audio, and actions |
| Monospaced values | IDs, timestamps, scores and model versions where character alignment helps |

The interface uses real status text alongside color. Buttons and inputs retain visible focus indicators. Empty states tell the user what action is available; disabled analysis states explain that the separate GTM model is required. Audio capture must never begin until consent is explicitly acknowledged. The upload form accepts multiple files and reports each result separately.

## Reusable patterns

- **Page header:** short title, one-sentence purpose, then the next useful action.
- **Metric cards:** one prominent value and a specific label; no invented activity or performance number.
- **Evidence panels:** model A, model B, agreement, quality, and alert/review status are distinct so disagreement cannot be hidden in a single score.
- **Forms:** label above control, inline validation, explicit success/error feedback, and disabled state only with an explanation.
- **Severity:** text and icon/shape with color; critical actions receive stronger contrast than ordinary events.
- **Tables and lists:** chronological ordering, readable timestamps, accessible row actions and honest no-results messages.

## Review checklist

Verify the login, dashboard, upload, live, event detail, alerts, review queue, reports, and admin pages at 390px, 768px and 1440px widths. Check keyboard tab order, visible focus, form labels, contrast, text zoom, and `prefers-reduced-motion`. Visual browser review covered desktop login/dashboard/upload/live and 390px login/registration/upload/live; the full accessibility and remaining page/device checks remain listed in `TEST_PLAN.md` until executed.
