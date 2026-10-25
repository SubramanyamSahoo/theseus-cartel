# Research website

GitHub Pages serves `main:/docs`.

Edit `docs/index.html`, `docs/styles.css`, and `docs/app.js`, then push main.
The `.nojekyll` file keeps this a plain static site.
Existing protocol documents are preserved.
Data files in `docs/assets/` are snapshots of the recorded campaign analysis.
The PDF is an automatically generated draft requiring review.

## Reading enhancements

`enhancements.css` and `enhancements.js` add a reader guide, evidence explorer, theme preference, reading progress, section links, print shortcut, and BibTeX download. The explorer reads the recorded `assets/primary_effects.json`; it does not simulate outcomes. Existing tables remain usable if JavaScript or data loading fails.
