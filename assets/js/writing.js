/*
 * /blog/: filter the list by kind (All / Write-ups / News) and search it.
 * Without JS the controls stay hidden and every post is listed.
 * /blog/#writeups and /blog/#news open with that filter applied.
 */
(function () {
  'use strict';

  var list = document.getElementById('writing-list');
  if (!list) return;

  var items = Array.prototype.slice.call(list.querySelectorAll('.blog-item'));
  var buttons = Array.prototype.slice.call(document.querySelectorAll('.writing-filter'));
  var input = document.getElementById('search-input');
  var empty = document.getElementById('nofound');
  var HASH_TO_KIND = { writeups: 'writeup', news: 'news' };
  var KIND_TO_HASH = { writeup: '#writeups', news: '#news', all: '' };
  var filter = 'all';

  function apply() {
    var query = input ? input.value.trim().toLowerCase() : '';
    var shown = 0;
    items.forEach(function (item) {
      var kindOk = filter === 'all' || item.getAttribute('data-kind') === filter;
      var textOk = !query || item.getAttribute('data-search').toLowerCase().indexOf(query) !== -1;
      item.hidden = !(kindOk && textOk);
      if (!item.hidden) shown++;
    });
    if (empty) empty.hidden = shown !== 0;
    buttons.forEach(function (b) {
      b.setAttribute('aria-pressed', b.getAttribute('data-filter') === filter ? 'true' : 'false');
    });
  }

  function setFilter(kind, updateUrl) {
    filter = kind;
    if (updateUrl && window.history.replaceState) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search + KIND_TO_HASH[kind]);
    }
    apply();
  }

  buttons.forEach(function (b) {
    b.addEventListener('click', function () { setFilter(b.getAttribute('data-filter'), true); });
  });
  if (input) input.addEventListener('input', apply);

  document.querySelectorAll('.writing-filters, .search-article').forEach(function (el) { el.hidden = false; });

  function fromHash() {
    var kind = HASH_TO_KIND[window.location.hash.slice(1)];
    setFilter(kind || 'all', false);
  }
  window.addEventListener('hashchange', fromHash);
  fromHash();
})();
