
const nav = document.getElementById('nav');
const menu = document.getElementById('menu');
const main = document.querySelector('main');
const narrow = matchMedia('(max-width:960px)');

function toggle(open, focus=false) {
  document.body.classList.toggle('menu-open', open);
  menu.setAttribute('aria-expanded', open);
  nav.inert = narrow.matches && !open;
  main.inert = narrow.matches && open;
  if (open) nav.querySelector('a').focus();
  if (focus) menu.focus();
}
menu.onclick = () => toggle(!document.body.classList.contains('menu-open'));
document.querySelector('.shade').onclick = () => toggle(false, true);
nav.querySelectorAll('a').forEach(a => a.onclick = () => toggle(false));
narrow.addEventListener('change', () => toggle(false));
toggle(false);

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && document.body.classList.contains('menu-open')) {
    toggle(false, true);
  }
  if (e.key === 'Tab' && narrow.matches && document.body.classList.contains('menu-open')) {
    const items = [menu, ...nav.querySelectorAll('a')];
    if (e.shiftKey && document.activeElement === items[0]) {
      e.preventDefault();
      items.at(-1).focus();
    } else if (!e.shiftKey && document.activeElement === items.at(-1)) {
      e.preventDefault();
      items[0].focus();
    }
  }
});

document.querySelectorAll('[data-history]').forEach(button => {
  button.onclick = () => {
    const reset = button.dataset.history === 'reset';
    document.querySelectorAll('[data-history]').forEach(
      other => other.setAttribute('aria-pressed', other === button)
    );
    const history = document.getElementById('history');
    history.classList.toggle('reset', reset);
    history.textContent = reset
      ? 'Earlier observations are cleared for both agents at full replacement.'
      : 'Earlier prices and outcomes remain available after full replacement.';
  };
});

document.querySelectorAll('[data-filter]').forEach(button => {
  button.onclick = () => {
    let count = 0;
    document.querySelectorAll('#effects tr').forEach(row => {
      row.hidden = button.dataset.filter !== 'all'
        && row.dataset.regime !== button.dataset.filter;
      if (!row.hidden) count++;
    });
    document.querySelectorAll('[data-filter]').forEach(
      other => other.setAttribute('aria-pressed', other === button)
    );
    document.getElementById('table-status').textContent =
      'Showing ' + count + ' primary comparisons.';
  };
});

const dialog = document.querySelector('dialog');
document.querySelector('.zoom').onclick = () => dialog.showModal();
document.getElementById('close-figure').onclick = () => dialog.close();
dialog.onclick = e => { if (e.target === dialog) dialog.close(); };

document.getElementById('copy').onclick = async () => {
  const citation = document.getElementById('bib');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'Citation copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or Command+C.';
  }
};

const sections = [...document.querySelectorAll('article section')];
const links = [...document.querySelectorAll('nav a')];
let pending = false;
function update() {
  pending = false;
  let id = sections[0].id;
  for (const section of sections) {
    if (section.getBoundingClientRect().top < 180) id = section.id;
  }
  links.forEach(link => {
    const selected = link.hash === '#' + id;
    link.classList.toggle('active', selected);
    if (selected) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
addEventListener('scroll', () => {
  if (!pending) {
    pending = true;
    requestAnimationFrame(update);
  }
}, {passive:true});
update();
