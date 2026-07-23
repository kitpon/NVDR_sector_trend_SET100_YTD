#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns
import urllib.parse
import json

# Constants
DEFAULT_BASE_URL = "http://setsmart.uobkhth.co.th:9081/ism/nvdrHistorical.html"
DEFAULT_COOKIE_FILE = "session_cookie.txt"
DEFAULT_STOCK_LIST = "stock_list.txt"
DEFAULT_SECTOR_FILE = "set50_list.csv"
DEFAULT_LOGO_FILE = "UOBKH_logo.png"
DEFAULT_TOP_N = 10
DEFAULT_CHUNK_DAYS = 28

def parse_args():
    parser = argparse.ArgumentParser(description="NVDR Study Tool (Python Version)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--stock-list", default=DEFAULT_STOCK_LIST, help="Text file with one symbol per line")
    parser.add_argument("--sector-file", default=DEFAULT_SECTOR_FILE, help="CSV file with Symbol,Sector columns")
    parser.add_argument("--logo-file", default=DEFAULT_LOGO_FILE, help="Logo image used in infographic")
    parser.add_argument("--symbol", action="append", dest="symbol_list", default=[], help="Add one symbol")
    parser.add_argument("--symbols", dest="symbols_csv", help="Add multiple symbols (comma separated)")
    parser.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help="Cookie header text file")
    parser.add_argument("--output-dir", help="Output directory (default: NVDR_YYYYMMDD_YYYYMMDD)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Row count for top buy/sell tables")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel request count")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="Max calendar days per request window")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Override source URL")
    parser.add_argument("--logo", choices=["on", "off"], default="on", help="Show logo in report (default: on)")
    parser.add_argument("--sync", action="store_true", help="Incremental sync: only download data after the latest date in existing CSVs")

    args = parser.parse_args()

    all_symbols = args.symbol_list if args.symbol_list else []
    if args.symbols_csv:
        all_symbols.extend(args.symbols_csv.split(","))
    args.symbols = all_symbols
    
    # Normalize dates
    try:
        args.start = datetime.strptime(args.start, "%Y-%m-%d").strftime("%Y-%m-%d")
        args.end = datetime.strptime(args.end, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as e:
        parser.error(f"Invalid date format: {e}")

    if args.start > args.end:
        parser.error(f"--start must be on or before --end. Received {args.start} > {args.end}.")

    return args

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def compact_date(iso_date):
    return iso_date.replace("-", "")

def default_study_dir(start_iso, end_iso):
    return f"NVDR_{compact_date(start_iso)}_{compact_date(end_iso)}"

def read_symbols(stock_list_path, inline_symbols):
    symbols = []
    if os.path.exists(stock_list_path):
        with open(stock_list_path, "r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                symbols.append(value.upper())
    
    for s in inline_symbols:
        if s.strip():
            symbols.append(s.strip().upper())
    
    return sorted(list(set(symbols)))

def read_sector_map(csv_path):
    if not os.path.exists(csv_path):
        print(f"Warning: Sector file not found: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        if 'symbol' not in df.columns or 'sector' not in df.columns:
            print(f"Warning: Sector file must have Symbol and Sector columns: {csv_path}")
            return {}
        
        return dict(zip(df['symbol'].str.upper(), df['sector']))
    except Exception as e:
        print(f"Error reading sector file: {e}")
        return {}

def iso_to_site_date(iso_date):
    # YYYY-MM-DD -> DD/MM/YYYY
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    return dt.strftime("%d/%m/%Y")

def split_date_range(start_iso, end_iso, chunk_days):
    ranges = []
    start_dt = datetime.strptime(start_iso, "%Y-%m-%d")
    end_dt = datetime.strptime(end_iso, "%Y-%m-%d")
    
    cursor = start_dt
    while cursor <= end_dt:
        chunk_end = cursor + timedelta(days=chunk_days - 1)
        effective_end = min(chunk_end, end_dt)
        ranges.append((cursor.strftime("%Y-%m-%d"), effective_end.strftime("%Y-%m-%d")))
        cursor = effective_end + timedelta(days=1)
    return ranges

def parse_number(text):
    if not text:
        return 0.0
    cleaned = text.replace(",", "").strip()
    if cleaned == "" or cleaned == "-":
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def build_url(base_url, symbol, start_iso, end_iso, extra_params=None):
    site_start = iso_to_site_date(start_iso)
    site_end = iso_to_site_date(end_iso)
    params = {
        "symbol": symbol,
        "beginDate": site_start,
        "endDate": site_end,
        "showBeginDate": site_start,
        "showEndDate": site_end,
        "quickPeriod": "",
        "lstDisplay": "value",
        "lstPeriod": "D",
        "locale": "en_US",
    }
    if extra_params:
        params.update(extra_params)
    
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def parse_rows(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    table_rows = soup.find_all("tr", class_=["odd", "even"])
    if not table_rows:
        return []
        
    for tr in table_rows:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 6:
            continue
        
        try:
            date_iso = datetime.strptime(cells[0], "%d/%m/%Y").strftime("%Y-%m-%d")
            rows.append({
                "date": date_iso,
                "buy_mb": parse_number(cells[1]),
                "sell_mb": parse_number(cells[2]),
                "buy_sell_mb": parse_number(cells[3]),
                "net_mb": parse_number(cells[4]),
                "nvdr_pct": parse_number(cells[5]),
            })
        except ValueError:
            continue
    return rows

def parse_total_row(html):
    soup = BeautifulSoup(html, "html.parser")
    total_td = soup.find("td", class_="table", string=lambda s: s and "Total" in s)
    if not total_td:
        table_tds = soup.find_all("td", class_="table")
        for td in table_tds:
            if td.find("strong", string="Total"):
                total_td = td
                break

    if not total_td:
        return None
    
    tr = total_td.parent
    cells = [td.get_text(strip=True) for td in tr.find_all("td")]
    if len(cells) < 6:
        return None
    
    return {
        "buy_mb": parse_number(cells[1]),
        "sell_mb": parse_number(cells[2]),
        "buy_sell_mb": parse_number(cells[3]),
        "net_mb": parse_number(cells[4]),
        "nvdr_pct": parse_number(cells[5]),
    }

def fetch_symbol_page(session, symbol, base_url, cookie_header, start_iso, end_iso, extra_params=None):
    url = build_url(base_url, symbol, start_iso, end_iso, extra_params)
    headers = {
        "Cookie": cookie_header,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/3.x NVDRStudy",
    }
    response = session.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    html = response.text
    rows = parse_rows(html)
    total = parse_total_row(html)
    return {"url": url, "rows": rows, "total": total}

def fetch_symbol(symbol, args, cookie_header, raw_dir):
    ranges = split_date_range(args.start, args.end, args.chunk_days)
    all_rows = []
    
    csv_path = os.path.join(raw_dir, f"{symbol}_nvdr_value.csv")
    existing_df = None
    last_date = None
    
    if args.sync and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                existing_df['date'] = existing_df['date'].astype(str)
                last_date = existing_df['date'].max()
                if last_date >= args.end:
                    return {"symbol": symbol, "rows": existing_df.to_dict('records'), "url": "", "total": None}
                
                sync_start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                ranges = split_date_range(sync_start, args.end, args.chunk_days)
        except Exception as e:
            print(f"  {symbol}: Sync error, full fetch. {e}")

    session = requests.Session()
    last_url = ""
    last_total = None
    
    for r_start, r_end in ranges:
        result = fetch_symbol_page(session, symbol, args.base_url, cookie_header, r_start, r_end)
        all_rows.extend(result["rows"])
        last_url = result["url"]
        if result["total"]:
            last_total = result["total"]
    
    if existing_df is not None:
        new_rows_df = pd.DataFrame(all_rows)
        combined_df = pd.concat([existing_df, new_rows_df])
        combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date')
        unique_rows = combined_df.to_dict('records')
    else:
        seen_dates = set()
        unique_rows = []
        for row in sorted(all_rows, key=lambda x: x["date"]):
            if row["date"] not in seen_dates:
                seen_dates.add(row["date"])
                unique_rows.append(row)
    
    return { "symbol": symbol, "rows": unique_rows, "url": last_url, "total": last_total }

def export_web_data(all_symbol_data, sector_map, output_dir):
    web_data = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbols": list(sector_map.keys())
        },
        "stocks": {}
    }
    for df in all_symbol_data:
        symbol = df['symbol'].iloc[0]
        compact_df = df[['date', 'buy_mb', 'sell_mb', 'buy_sell_mb', 'net_mb']].copy()
        web_data["stocks"][symbol] = {
            "sector": sector_map.get(symbol, "Unknown"),
            "data": compact_df.to_dict('records')
        }
    with open(os.path.join(output_dir, "web_data.json"), "w", encoding="utf-8") as f:
        json.dump(web_data, f)

def generate_dashboard(output_dir, args):
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVDR Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tom-select@2.2.2/dist/css/tom-select.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/tom-select@2.2.2/dist/js/tom-select.complete.min.js"></script>
    <style>
        :root {{ --uob-blue: #0f4c81; --bg-gray: #f4f7fb; --border: #d6deea; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: var(--bg-gray); color: #1b2430; }}
        header {{ background: var(--uob-blue); color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .logo-box img {{ max-height: 50px; filter: brightness(0) invert(1); }}
        .container {{ padding: 20px; max-width: 1400px; margin: 0 auto; }}
        .controls {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; align-items: end; }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; font-size: 0.9rem; }}
        input, select {{ width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: 4px; box-sizing: border-box; }}
        button {{ background: var(--uob-blue); color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: bold; }}
        button:hover {{ opacity: 0.9; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .full-width {{ grid-column: 1 / -1; }}
        h2 {{ margin-top: 0; color: var(--uob-blue); font-size: 1.2rem; border-bottom: 2px solid var(--bg-gray); padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th, td {{ padding: 10px; border-bottom: 1px solid var(--bg-gray); text-align: center; }}
        th {{ background: #f8fafc; font-weight: bold; }}
        .num-pos {{ color: #059669; font-weight: bold; }}
        .num-neg {{ color: #dc2626; font-weight: bold; }}
        #chart-container {{ min-height: 500px; }}
        .loading-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.9); display: flex; justify-content: center; align-items: center; z-index: 1000; font-weight: bold; }}
    </style>
</head>
<body>
    <div id="loading" class="loading-overlay">Loading Web Data...</div>
    <header>
        <div>
            <h1 style="margin:0; font-size: 1.5rem;">NVDR Interactive Dashboard</h1>
            <div id="data-info" style="font-size: 0.8rem; opacity: 0.8;"></div>
        </div>
        <div class="logo-box">
             {f'<img src="{os.path.basename(args.logo_file)}">' if args.logo == "on" and os.path.exists(args.logo_file) else ""}
        </div>
    </header>

    <div class="container">
        <div class="controls">
            <div>
                <label>Start Date</label>
                <input type="date" id="start-date">
            </div>
            <div>
                <label>End Date</label>
                <input type="date" id="end-date">
            </div>
            <div>
                <label>Compare Stocks</label>
                <select id="stock-select" multiple placeholder="Select stocks..."></select>
            </div>
            <div>
                <label>Compare Sectors</label>
                <select id="sector-select" multiple placeholder="Select sectors..."></select>
            </div>
            <div style="display: flex; gap: 10px;">
                <button onclick="processData()" id="btn-go" style="flex: 2;">PROCESS & UPDATE</button>
                <button onclick="resetFilters()" style="flex: 1; background: #64748b;">RESET</button>
            </div>
        </div>

        <div class="grid">
            <div class="card full-width">
                <h2>Comparison Chart (Cumulative Net Flow)</h2>
                <div id="comparison-chart"></div>
            </div>

            <div class="card">
                <h2>Top NVDR Net Buy (Selected Period)</h2>
                <table id="table-buy">
                    <thead><tr><th>Symbol</th><th>Sector</th><th>NET (MB)</th><th>Turnover</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>

            <div class="card">
                <h2>Top NVDR Net Sell (Selected Period)</h2>
                <table id="table-sell">
                    <thead><tr><th>Symbol</th><th>Sector</th><th>NET (MB)</th><th>Turnover</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>

            <div class="card full-width">
                <h2>Sector Summary (Selected Period)</h2>
                <table id="table-sector">
                    <thead><tr><th>Sector</th><th>Symbols</th><th>Total Buy</th><th>Total Sell</th><th>Total NET (MB)</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let rawData = null;
        let stockSelect = null;
        let sectorSelect = null;

        async function init() {{
            try {{
                const response = await fetch('web_data.json');
                rawData = await response.json();
                document.getElementById('loading').style.display = 'none';
                document.getElementById('data-info').innerText = `Generated: ${{rawData.metadata.generated_at}} | Universe: ${{rawData.metadata.symbols.length}} symbols`;

                let allDates = [];
                Object.values(rawData.stocks).forEach(s => s.data.forEach(d => allDates.push(d.date)));
                allDates.sort();
                const minDate = allDates[0];
                const maxDate = allDates[allDates.length - 1];

                document.getElementById('start-date').value = minDate;
                document.getElementById('end-date').value = maxDate;

                const symbols = Object.keys(rawData.stocks).sort();
                const sectors = [...new Set(Object.values(rawData.stocks).map(s => s.sector))].sort();

                stockSelect = new TomSelect('#stock-select', {{ 
                    options: symbols.map(s => ({{ value: s, text: s }})),
                    plugins: ['remove_button']
                }});
                
                sectorSelect = new TomSelect('#sector-select', {{ 
                    options: sectors.map(s => ({{ value: s, text: s }})),
                    plugins: ['remove_button']
                }});

                processData();
            }} catch (e) {{
                console.error(e);
                alert("Failed to load web_data.json. Ensure it exists in the same folder.");
            }}
        }}

        function formatNum(n) {{
            return n.toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
        }}

        function resetFilters() {{
            stockSelect.clear();
            sectorSelect.clear();
            processData();
        }}

        function processData() {{
            const start = document.getElementById('start-date').value;
            const end = document.getElementById('end-date').value;
            if (!start || !end) return;

            const summaries = [];
            const sectorMap = {{}};

            Object.entries(rawData.stocks).forEach(([symbol, info]) => {{
                const filtered = info.data.filter(d => d.date >= start && d.date <= end);
                if (filtered.length === 0) return;

                const net = filtered.reduce((sum, d) => sum + d.net_mb, 0);
                const turnover = filtered.reduce((sum, d) => sum + d.buy_sell_mb, 0);
                const buy = filtered.reduce((sum, d) => sum + d.buy_mb, 0);
                const sell = filtered.reduce((sum, d) => sum + d.sell_mb, 0);

                summaries.push({{ symbol, sector: info.sector, net, turnover, buy, sell, data: filtered }});

                if (!sectorMap[info.sector]) {{
                    sectorMap[info.sector] = {{ buy: 0, sell: 0, net: 0, count: 0, symbols: new Set() }};
                }}
                sectorMap[info.sector].buy += buy;
                sectorMap[info.sector].sell += sell;
                sectorMap[info.sector].net += net;
                sectorMap[info.sector].count++;
                sectorMap[info.sector].symbols.add(symbol);
            }});

            updateTables(summaries, sectorMap);
            updateChart(summaries, sectorMap, start, end);
        }}

        function updateTables(summaries, sectorMap) {{
            const sorted = [...summaries].sort((a, b) => b.net - a.net);
            const topBuy = sorted.slice(0, 10);
            const topSell = sorted.slice(-10).reverse();

            const renderRows = (data, targetId) => {{
                const tbody = document.querySelector(`#${{targetId}} tbody`);
                tbody.innerHTML = data.map(r => `
                    <tr>
                        <td><strong>${{r.symbol}}</strong></td>
                        <td>${{r.sector}}</td>
                        <td class="${{r.net >= 0 ? 'num-pos' : 'num-neg'}}">${{formatNum(r.net)}}</td>
                        <td>${{formatNum(r.turnover)}}</td>
                    </tr>
                `).join('');
            }};

            renderRows(topBuy, 'table-buy');
            renderRows(topSell, 'table-sell');

            const sectorTable = document.querySelector('#table-sector tbody');
            sectorTable.innerHTML = Object.entries(sectorMap)
                .sort((a, b) => b[1].net - a[1].net)
                .map(([name, data]) => `
                    <tr>
                        <td><strong>${{name}}</strong></td>
                        <td>${{data.symbols.size}}</td>
                        <td>${{formatNum(data.buy)}}</td>
                        <td>${{formatNum(data.sell)}}</td>
                        <td class="${{data.net >= 0 ? 'num-pos' : 'num-neg'}}">${{formatNum(data.net)}}</td>
                    </tr>
                `).join('');
        }}

        function updateChart(summaries, sectorMap, start, end) {{
            const selectedStocks = stockSelect.getValue();
            const selectedSectors = sectorSelect.getValue();
            const traces = [];

            let itemsToTrace = selectedStocks.length > 0 || selectedSectors.length > 0 
                ? summaries.filter(s => selectedStocks.includes(s.symbol))
                : summaries.sort((a, b) => b.net - a.net).slice(0, 5);

            itemsToTrace.forEach(s => {{
                let cumNet = 0;
                const traceData = s.data.map(d => {{
                    cumNet += d.net_mb;
                    return {{ x: d.date, y: cumNet }};
                }});
                traces.push({{
                    x: traceData.map(d => d.x),
                    y: traceData.map(d => d.y),
                    name: s.symbol,
                    type: 'scatter',
                    mode: 'lines+markers'
                }});
            }});

            selectedSectors.forEach(secName => {{
                const secData = {{}};
                summaries.filter(s => s.sector === secName).forEach(s => {{
                    s.data.forEach(d => {{
                        secData[d.date] = (secData[d.date] || 0) + d.net_mb;
                    }});
                }});
                const sortedDates = Object.keys(secData).sort();
                let cumNet = 0;
                const traceData = sortedDates.map(d => {{
                    cumNet += secData[d];
                    return {{ x: d, y: cumNet }};
                }});
                traces.push({{
                    x: traceData.map(d => d.x),
                    y: traceData.map(d => d.y),
                    name: `Sector: ${{secName}}`,
                    line: {{ width: 4, dash: 'dot' }},
                    type: 'scatter'
                }});
            }});

            const layout = {{
                margin: {{ t: 20, b: 40, l: 60, r: 20 }},
                hovermode: 'x unified',
                xaxis: {{ title: 'Date' }},
                yaxis: {{ title: 'Cumulative Net (MB)', tickformat: ',.0f' }},
                legend: {{ orientation: 'h', y: -0.2 }}
            }};
            Plotly.newPlot('comparison-chart', traces, layout, {{ responsive: true }});
        }}

        init();
    </script>
</body>
</html>"""
    with open(os.path.join(output_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html_template)

def main():
    args = parse_args()
    if not args.output_dir:
        args.output_dir = default_study_dir(args.start, args.end)
    
    raw_dir = os.path.join(args.output_dir, "raw")
    chart_dir = os.path.join(args.output_dir, "charts")
    report_dir = os.path.join(args.output_dir, "report")
    
    ensure_dir(args.output_dir)
    ensure_dir(raw_dir)
    ensure_dir(chart_dir)
    ensure_dir(report_dir)
    
    if not os.path.exists(args.cookie_file):
        print(f"Error: Cookie file not found: {args.cookie_file}")
        sys.exit(1)
        
    with open(args.cookie_file, "r", encoding="utf-8") as f:
        cookie_header = f.read().strip()
        
    sector_map = read_sector_map(args.sector_file)
    symbols = read_symbols(args.stock_list, args.symbols)
    
    if not symbols:
        print("Error: No symbols found.")
        sys.exit(1)
        
    print(f"Processing NVDR value history for {len(symbols)} symbols...")
    
    summaries = []
    failures = []
    
    def process_symbol(symbol):
        try:
            data = fetch_symbol(symbol, args, cookie_header, raw_dir)
            return {"ok": True, "data": data}
        except Exception as e:
            return {"ok": False, "symbol": symbol, "error": str(e)}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(process_symbol, symbols))
        
    all_symbol_data = []
    for res in results:
        if res["ok"]:
            data = res["data"]
            symbol = data["symbol"]
            if not data["rows"]: continue
            df = pd.DataFrame(data["rows"])
            df['symbol'] = symbol
            df['sector'] = sector_map.get(symbol, "Unknown")
            df = df.sort_values("date")
            df['cumulative_net_mb'] = df['net_mb'].cumsum()
            df.to_csv(os.path.join(raw_dir, f"{symbol}_nvdr_value.csv"), index=False)
            
            summary = {
                "symbol": symbol,
                "sector": df['sector'].iloc[0],
                "trading_days": len(df),
                "total_buy_mb": df['buy_mb'].sum(),
                "total_sell_mb": df['sell_mb'].sum(),
                "total_turnover_mb": df['buy_sell_mb'].sum(),
                "total_net_mb": df['net_mb'].sum(),
            }
            summaries.append(summary)
            all_symbol_data.append(df)
        else:
            failures.append({"symbol": res["symbol"], "error": res["error"]})

    if not summaries:
        print("No data collected.")
        return

    summaries_df = pd.DataFrame(summaries).sort_values("total_net_mb", ascending=False)
    summaries_df.to_csv(os.path.join(report_dir, "nvdr_summary.csv"), index=False)
    
    full_df = pd.concat(all_symbol_data)
    sector_daily = full_df.groupby(['sector', 'date']).agg({'buy_mb': 'sum', 'sell_mb': 'sum', 'buy_sell_mb': 'sum', 'net_mb': 'sum'}).reset_index()
    sector_daily = sector_daily.sort_values(['sector', 'date'])
    sector_daily['cumulative_net_mb'] = sector_daily.groupby('sector')['net_mb'].cumsum()
    sector_daily.to_csv(os.path.join(report_dir, "sector_nvdr_daily.csv"), index=False)
    
    sector_summary = sector_daily.groupby('sector').agg({'net_mb': ['sum', 'count'], 'buy_mb': 'sum', 'sell_mb': 'sum', 'buy_sell_mb': 'sum'}).reset_index()
    sector_summary.columns = ['sector', 'total_net_mb', 'trading_days', 'total_buy_mb', 'total_sell_mb', 'total_turnover_mb']
    
    sector_meta = []
    for sector in sector_summary['sector']:
        sym_count = len(full_df[full_df['sector'] == sector]['symbol'].unique())
        sector_meta.append({"sector": sector, "symbol_count": sym_count})
    
    sector_summary = sector_summary.merge(pd.DataFrame(sector_meta), on='sector')
    sector_summary.sort_values("total_net_mb", ascending=False).to_csv(os.path.join(report_dir, "sector_nvdr_summary.csv"), index=False)

    export_web_data(all_symbol_data, sector_map, args.output_dir)
    generate_dashboard(args.output_dir, args)

    # Static report charts (Old style fallback)
    print("Generating static charts...")
    plt.style.use('ggplot')
    for df in all_symbol_data:
        sym = df['symbol'].iloc[0]
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        plt.plot(pd.to_datetime(df['date']), df['cumulative_net_mb'], marker='o')
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        plt.title(f"{sym} Cumulative Net")
        plt.savefig(os.path.join(chart_dir, f"{sym}_cumulative_net.png"))
        plt.savefig(os.path.join(chart_dir, f"{sym}_cumulative_net.svg"))
        plt.close()

    # Sector Heatmap
    heatmap_data = sector_daily.pivot(index="sector", columns="date", values="net_mb").iloc[:, -10:]
    formatted_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y") for d in heatmap_data.columns]
    heatmap_data.columns = formatted_dates
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'format': mtick.StrMethodFormatter('{x:,.0f}')})
    plt.savefig(os.path.join(report_dir, "sector_nvdr_heatmap.png"))
    plt.close()

    # Minimal Static HTML
    top_buy = summaries_df.head(args.top_n)
    top_sell = summaries_df.tail(args.top_n).iloc[::-1]
    failures_html = pd.DataFrame(failures).to_html() if failures else "None"
    
    html_content = f"""<!DOCTYPE html><html><head><style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background: #0f4c81; color: white; }}
    .logo {{ position: absolute; top: 20px; right: 20px; max-height: 60px; }}
    </style></head><body>
    {f'<img src="{os.path.abspath(args.logo_file)}" class="logo">' if args.logo == "on" and os.path.exists(args.logo_file) else ""}
    <h1>NVDR Static Report</h1>
    <p>Go to <a href="dashboard.html"><b>Interactive Dashboard</b></a> for full features.</p>
    <h2>Top Buy</h2>{top_buy.to_html(index=False)}
    <h2>Top Sell</h2>{top_sell.to_html(index=False)}
    <h2>Heatmap</h2><img src="report/sector_nvdr_heatmap.png" style="max-width:100%">
    <h2>Failures</h2>{failures_html}
    </body></html>"""
    
    with open(os.path.join(args.output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nDone. Dashboard: {os.path.join(args.output_dir, 'dashboard.html')}")

if __name__ == "__main__":
    main()
