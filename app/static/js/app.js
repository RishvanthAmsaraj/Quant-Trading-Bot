/* ============================================================
 * app.js — Quant Trading Bot Web UI
 * ============================================================ */

let lastResults = null;
let lastTrades = null;
let lastPlots = null;

// ===== Theme toggle =====
(function initTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('theme-toggle');
  btn.textContent = html.getAttribute('data-theme') === 'dark' ? 'Light' : 'Dark';
  btn.addEventListener('click', () => {
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    btn.textContent = next === 'dark' ? 'Light' : 'Dark';
  });
})();

// ===== Strategy chip toggle =====
document.querySelectorAll('.strategy-chip').forEach(chip => {
  chip.addEventListener('click', (e) => {
    e.preventDefault();
    chip.classList.toggle('selected');
    const cb = chip.querySelector('input');
    cb.checked = chip.classList.contains('selected');
  });
});

// ===== Tabs =====
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    tab.classList.add('active');
    const target = document.getElementById('tab-' + tab.dataset.tab);
    if (target) target.classList.add('active');
  });
});

// ===== Run backtest =====
document.getElementById('run-btn').addEventListener('click', async () => {
  const tickers = document.getElementById('tickers').value;
  const period = document.getElementById('period').value;
  const strategies = [...document.querySelectorAll('#strategy-group .strategy-chip.selected')].map(c => c.dataset.value);
  const initialCapital = parseFloat(document.getElementById('capital').value);
  const positionSizing = document.getElementById('sizing').value;
  const stopLoss = parseFloat(document.getElementById('stop-loss').value) / 100;
  const takeProfit = parseFloat(document.getElementById('take-profit').value) / 100;

  if (strategies.length === 0) {
    showError('Select at least one strategy.');
    return;
  }

  showLoading();

  try {
    const res = await fetch('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tickers,
        period,
        strategies,
        initial_capital: initialCapital,
        position_sizing: positionSizing,
        stop_loss: stopLoss,
        take_profit: takeProfit,
      }),
    });
    const data = await res.json();
    if (!data.success) {
      showError(data.error || 'Backtest failed');
      return;
    }

    lastResults = data.results;
    lastTrades = data.trades;
    lastPlots = data.plots;

    renderResults(data);
  } catch (err) {
    showError('Network error: ' + err.message);
  }
});

// ===== Download CSV =====
document.getElementById('download-btn').addEventListener('click', async () => {
  if (!lastResults) return;
  try {
    const res = await fetch('/api/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ results: lastResults }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'backtest_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Download failed', err);
  }
});

// ===== Render helpers =====
function showLoading() {
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('loading').style.display = 'flex';
  document.getElementById('error-container').style.display = 'none';
  document.getElementById('results-content').style.display = 'none';
}

function showError(msg) {
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('loading').style.display = 'none';
  document.getElementById('results-content').style.display = 'none';
  const ec = document.getElementById('error-container');
  ec.style.display = 'block';
  ec.innerHTML = `<div class="alert-error">${escapeHtml(msg)}</div>`;
}

function showContent() {
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('loading').style.display = 'none';
  document.getElementById('error-container').style.display = 'none';
  document.getElementById('results-content').style.display = 'block';
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ===== Render results =====
function renderResults(data) {
  showContent();
  const results = data.results || [];
  const clean = results.filter(r => !r.error);
  if (clean.length === 0) {
    showError('All backtest runs failed. Check tickers and try again.');
    return;
  }

  // Summary metrics
  renderSummary(data.summary);

  // Table
  renderTable(clean);

  // Chart select
  const chartSelect = document.getElementById('chart-select');
  chartSelect.innerHTML = '';
  clean.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `${r.strategy} — ${r.ticker}`;
    chartSelect.appendChild(opt);
  });
  chartSelect.value = '0';
  chartSelect.onchange = () => renderChart(clean, chartSelect.value);

  // Trades select
  const tradesSelect = document.getElementById('trades-select');
  tradesSelect.innerHTML = '';
  clean.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `${r.strategy} — ${r.ticker}`;
    tradesSelect.appendChild(opt);
  });
  tradesSelect.value = '0';
  tradesSelect.onchange = () => renderTrades(tradesSelect.value);

  // Data ticker select
  const dataTickerSelect = document.getElementById('data-ticker-select');
  const tickers = [...new Set(clean.map(r => r.ticker))];
  dataTickerSelect.innerHTML = '';
  tickers.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    dataTickerSelect.appendChild(opt);
  });
  dataTickerSelect.onchange = () => renderRawData(dataTickerSelect.value);

  // Render first selections
  renderChart(clean, '0');
  renderTrades('0');
  renderRawData(tickers[0]);
}

// ===== Summary =====
function renderSummary(s) {
  const container = document.getElementById('summary-metrics');
  if (!s || !s.n_runs) {
    container.innerHTML = '';
    return;
  }

  container.innerHTML = `
    <div class="metric-card">
      <div class="label">Runs</div>
      <div class="value accent">${s.n_runs}</div>
    </div>
    <div class="metric-card">
      <div class="label">Avg Sharpe</div>
      <div class="value ${s.avg_sharpe >= 0 ? 'positive' : 'negative'}">${s.avg_sharpe}</div>
    </div>
    <div class="metric-card">
      <div class="label">Avg Return</div>
      <div class="value ${s.avg_return >= 0 ? 'positive' : 'negative'}">${s.avg_return}%</div>
    </div>
    <div class="metric-card">
      <div class="label">Best Sharpe</div>
      <div class="value ${s.best_sharpe.value >= 0 ? 'positive' : 'negative'}"><span class="main">${s.best_sharpe.value}</span><span class="sub">${s.best_sharpe.strategy} · ${s.best_sharpe.ticker}</span></div>
    </div>
    <div class="metric-card">
      <div class="label">Best Return</div>
      <div class="value ${s.best_return.value >= 0 ? 'positive' : 'negative'}"><span class="main">${s.best_return.value}%</span><span class="sub">${s.best_return.strategy} · ${s.best_return.ticker}</span></div>
    </div>
    <div class="metric-card">
      <div class="label">Most Trades</div>
      <div class="value accent"><span class="main">${s.most_trades.value}</span><span class="sub">${s.most_trades.strategy} · ${s.most_trades.ticker}</span></div>
    </div>
  `;
}

// ===== Table =====
function renderTable(results) {
  const cols = ['Strategy', 'Ticker', 'Total Return %', 'CAGR %', 'Sharpe', 'Sortino', 'Max DD %', 'Win Rate %', 'Profit Factor', 'Trades', 'Exposure %'];
  const keys = ['strategy', 'ticker', 'total_return', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'win_rate', 'profit_factor', 'n_trades', 'exposure'];

  const head = document.getElementById('table-head');
  head.innerHTML = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';

  const body = document.getElementById('table-body');
  body.innerHTML = results.map(r => {
    // 0 is neutral — only strictly positive/negative values get P&L colors
    const cls = v => v > 0 ? 'pos' : (v < 0 ? 'neg' : 'neutral');
    return '<tr>' + keys.map((k, i) => {
      let v = r[k];
      if (typeof v === 'number') {
        const isPct = cols[i].includes('%');
        const formatted = isPct || k === 'sharpe' || k === 'sortino' || k === 'profit_factor' ? v.toFixed(2) : v;
        return `<td class="${cls(v)}">${isPct ? formatted + '%' : formatted}</td>`;
      }
      return `<td>${v}</td>`;
    }).join('') + '</tr>';
  }).join('');
}

// ===== Chart =====
function renderChart(results, idx) {
  const r = results[parseInt(idx)];
  if (!r) return;

  const container = document.getElementById('plotly-container');

  // Check if we have a plot URL (existing HTML dashboard)
  const plotEntry = lastPlots ? lastPlots.find(p => p.strategy === r.strategy && p.ticker === r.ticker) : null;

  if (plotEntry && plotEntry.plot_url) {
    container.innerHTML = `<div class="plot-container"><iframe src="${plotEntry.plot_url}" title="Interactive Dashboard"></iframe></div>`;
    return;
  }

  // Fallback: render equity curve from data
  if (plotEntry && plotEntry.equity && plotEntry.equity.length > 0) {
    const eq = plotEntry.equity;
    const dates = eq.map(e => e.date);
    const values = eq.map(e => e.equity);

    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    const trace = {
      x: dates,
      y: values,
      type: 'scatter',
      mode: 'lines',
      name: `${r.strategy} — ${r.ticker}`,
      line: { color: dark ? '#d8b36a' : '#a8872f', width: 2 },
      fill: 'tozeroy',
      fillcolor: dark ? 'rgba(216, 179, 106, 0.08)' : 'rgba(168, 135, 47, 0.08)',
    };

    const layout = {
      title: { text: `${r.strategy} — ${r.ticker} Equity Curve`, font: { size: 14 } },
      xaxis: { title: 'Date', gridcolor: dark ? 'rgba(255,255,255,0.06)' : '#e5e3da' },
      yaxis: { title: 'Portfolio Value ($)', gridcolor: dark ? 'rgba(255,255,255,0.06)' : '#e5e3da' },
      margin: { t: 40, r: 20, b: 40, l: 60 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: 'Inter, sans-serif', size: 12, color: dark ? '#99a1ad' : '#6d727c' },
      hovermode: 'x unified',
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
    container.style.borderRadius = '10px';
    container.style.overflow = 'hidden';
  } else {
    container.innerHTML = '<div class="results-placeholder" style="padding:40px;"><p>No chart data available for this run.</p></div>';
  }
}

// ===== Trades =====
function renderTrades(idx) {
  const container = document.getElementById('trades-container');
  if (!lastTrades) {
    container.innerHTML = '<div class="results-placeholder" style="padding:30px;"><p>No trade data available.</p></div>';
    return;
  }

  const r = lastResults[parseInt(idx)];
  const key = `${r.strategy}_${r.ticker}`;
  const trades = lastTrades[key];

  if (!trades || trades.length === 0) {
    container.innerHTML = '<div class="results-placeholder" style="padding:30px;"><p>No trades were executed for this run.</p></div>';
    return;
  }

  container.innerHTML = `<p style="font-size:13px; color:var(--color-text-muted); margin-bottom:8px;">${trades.length} trade(s)</p>`;
  const list = document.createElement('div');
  trades.forEach(t => {
    const pnlClass = t.pnl >= 0 ? 'pos' : 'neg';
    const side = String(t.side || '').toLowerCase();
    const sideClass = side === 'buy' ? 'buy' : 'sell';
    list.innerHTML += `
      <div class="trade-entry">
        <div class="trade-header">
          <span><span class="side-badge ${sideClass}">${escapeHtml((t.side || '').toUpperCase())}</span>${escapeHtml(r.ticker)}</span>
          <span class="pnl ${pnlClass}">${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)} (${t.return_pct >= 0 ? '+' : ''}${t.return_pct.toFixed(2)}%)</span>
        </div>
        <div class="trade-detail">Entry: ${t.entry_date} @ $${t.entry_price} → Exit: ${t.exit_date} @ $${t.exit_price}</div>
      </div>
    `;
  });
  container.appendChild(list);
}

// ===== Raw Data =====
async function renderRawData(ticker) {
  const container = document.getElementById('data-table-wrapper');
  if (!ticker) {
    container.innerHTML = '<div class="results-placeholder" style="padding:30px;"><p>Run a backtest to see raw data.</p></div>';
    return;
  }

  container.innerHTML = '<div class="results-placeholder" style="padding:20px;"><p>Loading data…</p></div>';

  try {
    const res = await fetch(`/api/data/${ticker}?period=${document.getElementById('period').value}`);
    const data = await res.json();
    if (!data.success) {
      container.innerHTML = `<div class="alert-error">${escapeHtml(data.error)}</div>`;
      return;
    }

    const rows = data.data.slice(0, 200);
    const columns = data.columns.filter(c => c !== 'index');

    let html = '<table><thead><tr>' + columns.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(row => {
      html += '<tr>' + columns.map(c => {
        let v = row[c];
        if (typeof v === 'number') v = v.toFixed(2);
        return `<td>${escapeHtml(String(v))}</td>`;
      }).join('') + '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<div class="alert-error">Error: ${escapeHtml(err.message)}</div>`;
  }
}
