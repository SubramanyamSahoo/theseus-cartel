
(() => {
  'use strict';

  const root = document.documentElement;
  const themeButton = document.getElementById('theme-toggle');

  function setTheme(dark) {
    root.dataset.theme = dark ? 'dark' : 'light';
    themeButton.setAttribute('aria-pressed', String(dark));
  }

  setTheme(root.dataset.theme === 'dark');

  themeButton.onclick = () => {
    setTheme(root.dataset.theme !== 'dark');
    try {
      localStorage.setItem('theseus-theme', root.dataset.theme);
    } catch {}
  };

  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
    try {
      if (!localStorage.getItem('theseus-theme')) setTheme(event.matches);
    } catch {}
  });

  let printTheme;
  addEventListener('beforeprint', () => {
    printTheme = root.dataset.theme;
    setTheme(false);
  });
  addEventListener('afterprint', () => {
    if (printTheme) setTheme(printTheme === 'dark');
  });
  document.getElementById('print-article').onclick = () => window.print();

  const status = document.getElementById('share-status');
  let statusTimer;

  async function copyLink(url) {
    clearTimeout(statusTimer);
    try {
      await navigator.clipboard.writeText(url);
      status.textContent = 'Link copied.';
    } catch {
      status.textContent = 'Copy this link: ' + url;
    }
    statusTimer = setTimeout(() => { status.textContent = ''; }, 9000);
  }

  document.getElementById('share-article').onclick = () =>
    copyLink(location.origin + location.pathname);

  document.querySelectorAll('article section[id] > h2').forEach(heading => {
    const title = heading.textContent.trim();
    const button = document.createElement('button');
    button.className = 'section-copy';
    button.textContent = '#';
    button.title = 'Copy link to this section';
    button.setAttribute('aria-label', 'Copy link to ' + title);
    button.onclick = () => copyLink(
      location.origin + location.pathname + '#' + heading.parentElement.id
    );
    heading.append(button);
  });

  const progress = document.querySelector('.reading-progress span');
  let pending = false;

  function updateProgress() {
    pending = false;
    const total = document.documentElement.scrollHeight - innerHeight;
    const fraction = total > 0
      ? Math.max(0, Math.min(1, scrollY / total))
      : 0;
    progress.style.transform = 'scaleX(' + fraction + ')';
  }

  addEventListener('scroll', () => {
    if (!pending) {
      pending = true;
      requestAnimationFrame(updateProgress);
    }
  }, {passive:true});
  addEventListener('resize', updateProgress);
  updateProgress();

  const definitions = {
    history_reset: {
      label: 'Retained history − joint reset',
      unit: 'Difference in normalized profit gain',
      meaning: 'Positive values mean higher terminal profit gain with the inherited history retained.'
    },
    public_vs_private_first_price: {
      label: 'Public − private replacement notice',
      unit: 'Difference in normalized first price',
      meaning: 'Positive values mean a higher first successor price after a public notice than after private notices.'
    },
    public_vs_placebo_persistence: {
      label: 'Public − administrative notice',
      unit: 'Difference in normalized profit gain',
      meaning: 'Positive values mean higher terminal profit gain after a public notice than after the administrative placebo.'
    }
  };

  const explorer = document.getElementById('result-explorer');
  const select = document.getElementById('contrast-select');
  const format = value =>
    (value > 0 ? '+' : '') + value.toFixed(4).replace('-', '−');
  const plain = value => value.toFixed(4).replace('-', '−');

  function render(records) {
    const key = select.value;
    const definition = definitions[key];
    const rows = ['neutral', 'planted'].map(regime =>
      records.find(record =>
        record.hypothesis === key && record.regime === regime
      )
    );

    document.getElementById('contrast-meaning').textContent =
      definition.meaning;

    const rawLow = Math.min(0, ...rows.map(record => record.ci_low));
    const rawHigh = Math.max(0, ...rows.map(record => record.ci_high));
    const padding = (rawHigh - rawLow) * .12 || .1;
    const low = rawLow - padding;
    const high = rawHigh + padding;
    const x = value => 200 + (value - low) / (high - low) * 510;

    let svg =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 205" ' +
      'role="img" aria-labelledby="explorer-title explorer-desc">' +
      '<title id="explorer-title">' + definition.label + '</title>' +
      '<desc id="explorer-desc">Two paired effects with recorded adjusted ' +
      'bootstrap intervals. Exact numbers are shown below.</desc>';

    svg +=
      '<line class="zero" x1="' + x(0) + '" x2="' + x(0) +
      '" y1="30" y2="125"/>';

    rows.forEach((record, index) => {
      const y = 52 + index * 55;
      const color = index ? 'var(--chart-copper)' : 'var(--chart-blue)';
      const name = index ? 'Instructed founders' : 'Neutral founders';

      svg +=
        '<text x="182" y="' + (y + 5) +
        '" font-size="17" text-anchor="end">' + name + '</text>' +
        '<line x1="' + x(record.ci_low) +
        '" x2="' + x(record.ci_high) +
        '" y1="' + y + '" y2="' + y +
        '" stroke="' + color +
        '" stroke-width="4" stroke-linecap="round"/>' +
        '<circle cx="' + x(record.effect) + '" cy="' + y +
        '" r="6" fill="' + color +
        '" stroke="var(--surface)" stroke-width="2"/>';
    });

    svg += '<line class="axis" x1="200" x2="710" y1="137" y2="137"/>';

    for (let i = 0; i < 5; i++) {
      const value = low + (high - low) * i / 4;
      svg +=
        '<text x="' + x(value) +
        '" y="160" font-size="15" text-anchor="middle">' +
        value.toFixed(2).replace('-', '−') + '</text>';
    }

    svg +=
      '<text x="455" y="190" text-anchor="middle" font-size="16">' +
      definition.unit + '</text></svg>';

    document.getElementById('explorer-chart').innerHTML = svg;

    document.getElementById('estimate-pair').innerHTML =
      rows.map((record, index) =>
        '<div><div class="estimate-label ' +
        (index ? 'instructed' : '') + '">' +
        (index ? 'Instructed founders' : 'Neutral founders') +
        '</div><div class="estimate-value">' +
        format(record.effect) +
        '</div><div class="estimate-detail">Adjusted interval<br>[' +
        plain(record.ci_low) + ', ' + plain(record.ci_high) +
        ']<br>Holm p = ' + record.holm_p_value.toFixed(4) +
        '</div></div>'
      ).join('');

    const excludesZero = rows.some(record =>
      record.ci_low > 0 || record.ci_high < 0
    );
    const rejects = rows.some(record => record.holm_p_value < .05);

    document.getElementById('explorer-note').textContent = rejects
      ? 'At least one displayed comparison passes the .05 Holm-adjusted threshold. Interpret it within the scope of the protocol.'
      : excludesZero
        ? 'One bootstrap interval excludes zero, but neither Holm-adjusted test passes .05. The interval and test use different procedures.'
        : 'Neither Holm-adjusted test passes .05. This does not establish that the two interventions are equivalent.';

    document.getElementById('explorer-meta').textContent =
      'Per founder condition: ' + rows[0].matched_trials +
      ' matched trials · ' + rows[0].clusters +
      ' product clusters. Intervals use dataset-stratified product-cluster ' +
      'bootstrapping with Bonferroni tails. “Instructed” describes the ' +
      'founder prompt, not verified cartel formation.';

    updateProgress();
  }

  fetch('assets/primary_effects.json')
    .then(response => {
      if (!response.ok) throw Error('Data unavailable');
      return response.json();
    })
    .then(records => {
      if (!Array.isArray(records) || records.length !== 6) {
        throw Error('Unexpected evidence format');
      }

      for (const key of Object.keys(definitions)) {
        for (const regime of ['neutral', 'planted']) {
          const matches = records.filter(record =>
            record.hypothesis === key && record.regime === regime
          );
          if (matches.length !== 1) {
            throw Error('Missing or duplicate comparison');
          }

          const record = matches[0];
          const fields = [
            'effect', 'ci_low', 'ci_high',
            'holm_p_value', 'matched_trials', 'clusters'
          ];

          if (
            !record.available ||
            !fields.every(field =>
              typeof record[field] === 'number' &&
              Number.isFinite(record[field])
            ) ||
            record.ci_low > record.ci_high ||
            record.holm_p_value < 0 ||
            record.holm_p_value > 1
          ) {
            throw Error('Invalid comparison');
          }
        }
      }

      select.onchange = () => render(records);
      explorer.hidden = false;
      render(records);
    })
    .catch(() => {
      document.getElementById('explorer-load-note').hidden = false;
    });
})();
