// Copy BibTeX to clipboard (from the Academic Project Page Template;
// the CSS appends "ied!" to "Cop" while the button is in the copied state)
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    function done() {
        button.classList.add('copied');
        copyText.textContent = 'Cop';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    }

    navigator.clipboard.writeText(bibtexElement.textContent).then(done).catch(function() {
        const textArea = document.createElement('textarea');
        textArea.value = bibtexElement.textContent;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        done();
    });
}

// Scroll to top
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    scrollButton.classList.toggle('visible', window.pageYOffset > 300);
});

// Rollout videos play only while visible, so the gallery never decodes 40 videos at once
const videoObserver = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
        const video = entry.target;
        if (entry.isIntersecting) {
            video.play().catch(function() {});
        } else {
            video.pause();
        }
    });
}, { threshold: 0.25 });

// Suite tabs of the rollout gallery
function showSuite(suite) {
    document.querySelectorAll('.suite-tabs li').forEach(function(tab) {
        tab.classList.toggle('is-active', tab.dataset.suite === suite);
    });
    document.querySelectorAll('.suite-panel').forEach(function(panel) {
        panel.hidden = panel.dataset.suite !== suite;
    });
}

// ---------------------------------------------------------------------------
// Charts. Both are rows of markers on a shared success-rate axis.

// Table 1 of the paper, ordered by LIBERO-Plus success.
// Drops are the paper's, computed before rounding.
const BENCHMARK_DATA = [
    { name: 'π₀.₅', drop: 12.2, libero: 97.7, plus: 85.5, knn: false },
    { name: 'MolmoAct2', drop: 14.9, libero: 97.5, plus: 82.6, knn: false },
    { name: 'k-NN (multimodal)', drop: 9.3, libero: 91.7, plus: 82.4, knn: true, best: true },
    { name: 'k-NN (LAC key)', drop: 11.3, libero: 93.0, plus: 81.7, knn: true },
    { name: 'GR00T N1.7', drop: 14.6, libero: 94.9, plus: 80.2, knn: false },
    { name: 'k-NN (ResNet)', drop: 13.2, libero: 88.7, plus: 75.5, knn: true },
    { name: 'VLA-JEPA', drop: 23.1, libero: 97.2, plus: 74.0, knn: false },
    { name: 'Evo-1', drop: 24.0, libero: 95.4, plus: 71.4, knn: false },
    { name: 'X-VLA', drop: 30.0, libero: 94.5, plus: 64.5, knn: false },
    { name: 'π₀', drop: 29.0, libero: 71.9, plus: 42.9, knn: false },
    { name: 'SmolVLA', drop: 54.3, libero: 82.2, plus: 27.9, knn: false },
    { name: 'π₀-FAST', drop: 64.8, libero: 90.6, plus: 25.7, knn: false },
];

// Table 2 of the paper: k-NN on each VLA's backbone before and after
// LIBERO fine-tuning, and the full VLA. GR00T freezes its VLM, so before = after.
const BACKBONE_DATA = [
    { name: 'π₀.₅', libero: [85.4, 81.8, 97.7], plus: [60.5, 65.7, 85.5] },
    { name: 'MolmoAct2', libero: [78.6, 91.4, 97.5], plus: [52.6, 81.0, 82.6] },
    { name: 'GR00T N1.7', frozen: true, libero: [87.6, 87.6, 94.9], plus: [66.4, 66.4, 80.2] },
    { name: 'VLA-JEPA', libero: [84.0, 83.0, 97.2], plus: [49.3, 65.7, 74.0] },
    { name: 'Evo-1', libero: [62.1, 87.1, 95.4], plus: [35.5, 72.6, 71.4] },
    { name: 'X-VLA', libero: [83.3, 88.6, 94.5], plus: [62.4, 78.1, 64.5] },
    { name: 'π₀', libero: [86.9, 17.9, 71.9], plus: [56.7, 2.4, 42.9] },
    { name: 'SmolVLA', libero: [76.3, 86.5, 82.2], plus: [45.0, 69.3, 27.9] },
    { name: 'π₀-FAST', libero: [86.6, 95.0, 90.6], plus: [58.3, 74.0, 25.7] },
];

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

// Each row: { name, rowClass, line: [a, b], marks: [{ cls, value }], delta, label, tipLead, tipSub }
function renderRows(chart, rows, axis) {
    const list = chart.querySelector('.db-rows');
    const tooltip = chart.querySelector('.db-tooltip');
    const pct = function(v) { return (v - axis.min) / (axis.max - axis.min) * 100; };
    list.replaceChildren();
    tooltip.hidden = true;

    rows.forEach(function(r) {
        const row = el('div', 'db-row ' + r.rowClass);
        row.setAttribute('role', 'listitem');
        row.tabIndex = 0;
        row.setAttribute('aria-label', r.label);

        const track = el('div', 'db-track');
        const lo = Math.min(r.line[0], r.line[1]);
        const hi = Math.max(r.line[0], r.line[1]);
        const line = el('span', 'db-line');
        line.style.left = pct(lo) + '%';
        line.style.width = (pct(hi) - pct(lo)) + '%';
        track.append(line);
        r.marks.forEach(function(m) {
            const mark = el('span', 'db-dot ' + m.cls);
            mark.style.left = pct(m.value) + '%';
            track.append(mark);
        });

        row.append(el('span', 'db-name', r.name), track, el('span', 'db-delta', r.delta));
        list.append(row);

        function show() {
            tooltip.replaceChildren(el('strong', null, r.tipLead), el('div', 'tt-name', r.tipSub));
            tooltip.hidden = false;
            const chartBox = chart.getBoundingClientRect();
            const trackBox = track.getBoundingClientRect();
            const rowBox = row.getBoundingClientRect();
            const mid = trackBox.left - chartBox.left + trackBox.width * (pct(lo) + pct(hi)) / 200;
            const left = Math.max(4, Math.min(mid - tooltip.offsetWidth / 2, chartBox.width - tooltip.offsetWidth - 4));
            tooltip.style.left = left + 'px';
            tooltip.style.top = (rowBox.top - chartBox.top - tooltip.offsetHeight - 6) + 'px';
        }
        function hide() { tooltip.hidden = true; }
        row.addEventListener('pointerenter', show);
        row.addEventListener('pointerleave', hide);
        row.addEventListener('focus', show);
        row.addEventListener('blur', hide);
    });
}

function renderBenchmarkChart() {
    const chart = document.getElementById('benchmark-chart');
    if (!chart) return;
    renderRows(chart, BENCHMARK_DATA.map(function(d) {
        const drop = d.drop.toFixed(1);
        return {
            name: d.name,
            rowClass: (d.knn ? 'is-knn' : 'is-vla') + (d.best ? ' is-best' : ''),
            line: [d.plus, d.libero],
            marks: [{ cls: 'is-filled', value: d.plus }, { cls: 'is-ring', value: d.libero }],
            delta: '−' + drop + ' %pt',
            label: d.name + ': LIBERO ' + d.libero + ' %, LIBERO-Plus ' + d.plus + ' %, difference ' + drop + ' percentage points',
            tipLead: d.libero + ' % → ' + d.plus + ' %',
            tipSub: d.name + ' · Δ −' + drop + ' %pt',
        };
    }), { min: 20, max: 100 });
}

function renderBackboneChart(benchmark) {
    const chart = document.getElementById('backbone-chart');
    if (!chart) return;
    const benchName = benchmark === 'plus' ? 'LIBERO-Plus' : 'LIBERO';
    renderRows(chart, BACKBONE_DATA.map(function(d) {
        const v = d[benchmark];
        const before = v[0], after = v[1], vla = v[2];
        const gap = after - vla;
        const gapText = (gap >= 0 ? '+' : '−') + Math.abs(gap).toFixed(1);
        return {
            name: d.name,
            rowClass: 'is-backbone' + (gap > 0 ? ' is-best' : ''),
            line: [before, after],
            marks: [
                { cls: 'is-ring', value: before },
                { cls: 'is-filled', value: after },
                { cls: 'is-vla-mark', value: vla },
            ],
            delta: gapText,
            label: d.name + ' on ' + benchName + ': k-NN before fine-tuning ' + before + ' %, after ' + after
                + ' %, VLA ' + vla + ' %',
            tipLead: 'k-NN ' + before + ' % → ' + after + ' %',
            tipSub: d.name + ' · VLA ' + vla + ' %' + (d.frozen ? ' · VLM frozen' : ''),
        };
    }), { min: 0, max: 100 });

    chart.querySelectorAll('.chart-toggle button').forEach(function(button) {
        button.classList.toggle('is-active', button.dataset.benchmark === benchmark);
        button.setAttribute('aria-pressed', button.dataset.benchmark === benchmark);
    });
}

// Pipeline figure: hovering (or tapping) one path grays out the other
function setupPipeline() {
    const canvas = document.getElementById('pipeline');
    if (!canvas) return;
    const canHover = window.matchMedia('(hover: hover)').matches;
    function focusPath(path) {
        canvas.classList.toggle('focus-knn', path === 'knn');
        canvas.classList.toggle('focus-vla', path === 'vla');
    }
    canvas.querySelectorAll('.path-zone:not(.is-part)').forEach(function(zone) {
        const path = zone.dataset.path;
        zone.addEventListener('pointerenter', function() { if (canHover) focusPath(path); });
        zone.addEventListener('pointerleave', function() { if (canHover) focusPath(null); });
        zone.addEventListener('focus', function() { focusPath(path); });
        zone.addEventListener('blur', function() { focusPath(null); });
        zone.addEventListener('click', function() {
            if (!canHover) focusPath(canvas.classList.contains('focus-' + path) ? null : path);
        });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    setupPipeline();
    renderBenchmarkChart();
    renderBackboneChart('plus');
    document.querySelectorAll('#backbone-chart .chart-toggle button').forEach(function(button) {
        button.addEventListener('click', function() { renderBackboneChart(button.dataset.benchmark); });
    });
    document.querySelectorAll('video.rollout').forEach(function(video) {
        videoObserver.observe(video);
    });
    document.querySelectorAll('.suite-tabs li').forEach(function(tab) {
        tab.querySelector('a').addEventListener('click', function(event) {
            event.preventDefault();
            showSuite(tab.dataset.suite);
        });
    });
});
