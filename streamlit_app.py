# streamlit_app.py
# ------------------------------------------------------------
# Portfolio dashboard (Streamlit)
# - Holdings avec:
#   * Gain/Loss TD-like (montant + % since-buy) sur 2 lignes (HTML)
#   * Return % (Since buy)   [indépendant de la plage]
#   * Return % (Window)      [dépend de la plage]
# - KPIs: valeur titres, cash, total, delta sur la fenêtre, perf totale depuis achat
# - Courbe NAV:
#   * Survol continu: affiche la valeur du portefeuille au x courant (capture sur toute la zone du graphique)
#   * Axe Y serré [min..max] + padding
# - Tableau: 5 premières lignes visibles, « Voir plus » à droite du titre
#   colonnes auto-ajustées (table-layout:auto), zébrage 1/2
#
# Requirements:
#   pip install streamlit yfinance pandas numpy pytz altair
#
# Run:
#   streamlit run streamlit_app.py
# ------------------------------------------------------------

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import streamlit as st
import altair as alt

APP_TZ = pytz.timezone("America/Toronto")
DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = datetime.now(APP_TZ).date()
DEFAULT_CASH = 8420.02

# -------- ordre d'origine (sera respecté dans le tableau) --------
DEFAULT_POSITIONS = [
    {"ticker": "ATD.TO",    "quantity": 18,  "avg_cost": 42.795},
    {"ticker": "ZIN.TO",    "quantity": 65,  "avg_cost": 32.0858},
    {"ticker": "BCE.TO",    "quantity": 58,  "avg_cost": 54.689},
    {"ticker": "BLX.TO",    "quantity": 98,  "avg_cost": 34.8919},
    {"ticker": "BIP-UN.TO", "quantity": 31,  "avg_cost": 46.4771},
    {"ticker": "CNR.TO",    "quantity": 13,  "avg_cost": 161.2985},
    {"ticker": "CTC-A.TO",  "quantity": 8,   "avg_cost": 191.9688},
    {"ticker": "CVE.TO",    "quantity": 24,  "avg_cost": 22.7763},
    {"ticker": "DOL.TO",    "quantity": 23,  "avg_cost": 79.7013},
    {"ticker": "FTS.TO",    "quantity": 32,  "avg_cost": 55.6922},
    {"ticker": "HXE.TO",    "quantity": 277, "avg_cost": 14.59},
    {"ticker": "IAG.TO",    "quantity": 62,  "avg_cost": 85.3211},
    {"ticker": "IVN.TO",    "quantity": 80,  "avg_cost": 18.4549},
    {"ticker": "XFN.TO",    "quantity": 297, "avg_cost": 42.5474},
    {"ticker": "XIT.TO",    "quantity": 112, "avg_cost": 39.9416},
    {"ticker": "XRE.TO",    "quantity": 137, "avg_cost": 14.6526},
    {"ticker": "XMA.TO",    "quantity": 266, "avg_cost": 14.6098},
    {"ticker": "MFC.TO",    "quantity": 213, "avg_cost": 23.5269},
    {"ticker": "MRU.TO",    "quantity": 15,  "avg_cost": 55.516},
    {"ticker": "NGT.TO",    "quantity": 37,  "avg_cost": 53.94},
    {"ticker": "NPI.TO",    "quantity": 32,  "avg_cost": 30.5622},
    {"ticker": "NTR.TO",    "quantity": 23,  "avg_cost": 108.2243},
    {"ticker": "QSR.TO",    "quantity": 10,  "avg_cost": 87.499},
    {"ticker": "TIXT.TO",   "quantity": 75,  "avg_cost": 40.5432},
    {"ticker": "TIH.TO",    "quantity": 15,  "avg_cost": 113.846},
    {"ticker": "TOU.TO",    "quantity": 38,  "avg_cost": 76.2429},
    {"ticker": "WCN.TO",    "quantity": 20,  "avg_cost": 129.2695},
]

# ---------- helpers ----------

def _normalize_positions_keep_order(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les colonnes et conserve l'ordre d'apparition."""
    df = df.reset_index(drop=False).rename(columns={"index": "order"})
    rename_map = {}
    for want in ["ticker", "quantity", "avg_cost", "buy_price"]:
        for c in df.columns:
            if c.lower() == want:
                rename_map[c] = want
    df = df.rename(columns=rename_map)
    if "avg_cost" not in df.columns and "buy_price" in df.columns:
        df = df.rename(columns={"buy_price": "avg_cost"})
    if "avg_cost" not in df.columns:
        df["avg_cost"] = np.nan

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    df = df[df["quantity"] > 0].copy()

    first_order = df.groupby("ticker", sort=False)["order"].min()
    agg = df.groupby("ticker", sort=False).agg(
        quantity=("quantity", "sum"),
        avg_cost=("avg_cost", "mean"),
    )
    agg["order"] = first_order
    agg = agg.reset_index().sort_values("order", kind="stable").reset_index(drop=True)
    return agg[["ticker", "quantity", "avg_cost", "order"]]


@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_prices(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    if len(tickers) == 0:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # inclure la fin
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            if t in data.columns.get_level_values(0):
                out[t] = data[t]["Close"]
        if not out:
            return pd.DataFrame()
        px = pd.concat(out, axis=1)
    else:
        px = data["Close"].to_frame(tickers[0])
    px = px.sort_index().dropna(how="all")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def latest_prices(px: pd.DataFrame) -> pd.Series:
    if px.empty:
        return pd.Series(dtype=float)
    return px.dropna(how="all").iloc[-1]


def start_end_prices(px: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if px.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    px_ff = px.ffill().bfill()
    return px_ff.iloc[0], px_ff.iloc[-1]


def portfolio_time_series(px: pd.DataFrame, qty_map: dict, cash: float = 0.0, include_cash: bool = False) -> pd.Series:
    if px.empty:
        return pd.Series(dtype=float)
    qty = pd.Series(qty_map, dtype=float).reindex(px.columns).fillna(0.0)
    nav = px.mul(qty, axis=1).sum(axis=1).rename("Portfolio (CAD)")
    return nav + float(cash) if include_cash else nav


def fmt_cur(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)


# ---------- UI ----------

st.set_page_config(page_title="Portefeuille Polyfinances", layout="wide")
st.title("Portefeuille Polyfinances")
st.subheader("Théo Gachet - VP Investissement")

with st.sidebar:
    
    # Dans ton bloc `with st.sidebar:` AVANT st.header("Inputs"), ajoute :

    # --- Logo PolyFinances centré en haut de la sidebar ---
    from pathlib import Path

    # essaie plusieurs extensions d'images courantes
    for _logo_name in ["polyfinances.png", "polyfinances.jpg", "polyfinances.jpeg", "polyfinances.svg"]:
        if Path(_logo_name).exists():
            col_l, col_mid, col_r = st.columns([1, 4, 1])
            with col_mid:
                st.image(_logo_name, use_container_width=True)
            break
    # ------------------------------------------------------

    st.header("Inputs")


    cash_balance = st.number_input(
        "Cash balance (CAD)",
        min_value=0.0,
        value=float(DEFAULT_CASH),
        step=100.0,
        help="Included in total portfolio value and % weights.",
    )

    preset = st.selectbox(
        "Quick range",
        ["Custom", "Last 1M", "Last 3M", "YTD", "Last 6M", "Last 12M"],
        index=2
    )

    today = DEFAULT_END
    if preset == "Last 1M":
        s_default = today - timedelta(days=30)
    elif preset == "Last 3M":
        s_default = today - timedelta(days=90)
    elif preset == "YTD":
        s_default = date(today.year, 1, 1)
    elif preset == "Last 6M":
        s_default = today - timedelta(days=182)
    elif preset == "Last 12M":
        s_default = today - timedelta(days=365)
    else:
        s_default = DEFAULT_START

    start_date, end_date = st.date_input(
        "Date range",
        value=(s_default, today),
        min_value=date(2000, 1, 1),
        max_value=today,
        help="Used for historical time series and window returns.",
    )

    include_cash_ts = st.checkbox(
        "Include cash in time series",
        value=True,  # cochée par défaut
        help="Adds a constant cash line to NAV over time.",
    )

# -------- positions + ordre --------

positions = pd.DataFrame(DEFAULT_POSITIONS)
positions["order"] = np.arange(len(positions), dtype=int)

if positions.empty:
    st.warning("No positions to display.")
    st.stop()

tickers = positions["ticker"].tolist()

# -------- prix --------
px = fetch_prices(tickers, start_date, end_date)
if px.empty:
    st.error("No price data returned. Check tickers or date range.")
    st.stop()

last_close = latest_prices(px)
start_prices, end_prices = start_end_prices(px)

qty_map = dict(zip(positions["ticker"], positions["quantity"]))

# -------- calculs --------
positions = positions.copy()
positions["Price"] = positions["ticker"].map(last_close.to_dict())
positions["Start Price"] = positions["ticker"].map(start_prices.to_dict())
positions["End Price"] = positions["ticker"].map(end_prices.to_dict())

positions["Mkt Value"] = positions["quantity"] * positions["Price"]
positions["Book Cost"] = positions["quantity"] * positions["avg_cost"]
positions["Gain/Loss Unrealized"] = positions["Mkt Value"] - positions["Book Cost"]

with np.errstate(divide="ignore", invalid="ignore"):
    positions["Return % (Since buy)"] = np.where(
        positions["avg_cost"] > 0,
        positions["Price"] / positions["avg_cost"] - 1.0,
        np.nan,
    )
with np.errstate(divide="ignore", invalid="ignore"):
    positions["Return % (Window)"] = positions["End Price"] / positions["Start Price"] - 1.0

total_investments = float(positions["Mkt Value"].sum())
total_value = total_investments + float(cash_balance)
positions["% of Portfolio"] = positions["Mkt Value"] / total_value

# -------- NAV séries + KPIs --------
nav_ex_cash = portfolio_time_series(px, qty_map, cash_balance, include_cash=False)
nav_incl_cash = portfolio_time_series(px, qty_map, cash_balance, include_cash=True)

if len(nav_incl_cash) >= 2 and nav_incl_cash.iloc[0] > 0:
    total_delta_window = nav_incl_cash.iloc[-1] / nav_incl_cash.iloc[0] - 1.0
else:
    total_delta_window = np.nan

total_cost = float(positions["Book Cost"].sum())
total_gain = total_investments - total_cost
total_perf_since_buy = (total_gain / total_cost) if total_cost > 0 else np.nan

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Securities value (CAD)", fmt_cur(total_investments))
kpi2.metric("Cash (CAD)", fmt_cur(cash_balance))
kpi3.metric(
    "Total portfolio (CAD)",
    fmt_cur(total_value),
    delta=f"{total_delta_window:.2%}" if pd.notna(total_delta_window) else None,
)
kpi4.metric(
    "Performance totale depuis achat",
    f"{total_perf_since_buy:.2%}" if pd.notna(total_perf_since_buy) else "n/a",
    delta=f"{total_gain:+,.2f} CAD"
)

# -------- tableau HTML: colonnes auto, zébrage, une seule table + toggle « Voir plus » --------
def color(v: float) -> str:
    return "green" if pd.notna(v) and v >= 0 else "red"

view = pd.DataFrame({
    "Ticker": positions["ticker"],
    "Quantity": positions["quantity"].map(lambda x: f"{x:,.0f}"),
    "Price": positions["Price"].map(fmt_cur),
    "Avg Cost": positions["avg_cost"].map(fmt_cur),
    "Mkt Value": positions["Mkt Value"].map(fmt_cur),
    "Book Cost": positions["Book Cost"].map(fmt_cur),
    "% of Portfolio": positions["% of Portfolio"].map(lambda x: f"{x:.2%}"),
    "Start Price": positions["Start Price"].map(fmt_cur),
    "End Price": positions["End Price"].map(fmt_cur),
    "Return % (Window)": positions["Return % (Window)"].map(
        lambda x: f"<span style='color:{color(x)}'>{x:.2%}</span>" if pd.notna(x) else "n/a"
    ),
    "Return % (Since buy)": positions["Return % (Since buy)"].map(
        lambda x: f"<span style='color:{color(x)}'>{x:.2%}</span>" if pd.notna(x) else "n/a"
    ),
})

gl_amt = positions["Gain/Loss Unrealized"]
gl_pct = positions["Return % (Since buy)"]
view["Gain/Loss (TD)"] = [
    f"<span style='color:{color(a)}'><b>{a:+,.2f}</b><br><span style='font-size:90%'>{(p if pd.notna(p) else float('nan')):+.2%}</span></span>"
    if pd.notna(a) and pd.notna(p) else
    (f"<span style='color:{color(a)}'><b>{a:+,.2f}</b><br><span style='font-size:90%'>n/a</span></span>" if pd.notna(a) else "n/a")
    for a, p in zip(gl_amt, gl_pct)
]

view = view[[
    "Ticker", "Quantity", "Price", "Avg Cost",
    "Mkt Value", "Book Cost", "Gain/Loss (TD)",
    "% of Portfolio", "Start Price", "End Price",
    "Return % (Window)", "Return % (Since buy)",
]]

# conserver l'ordre d’origine
view.insert(0, "order", positions["order"].values)
view = view.sort_values("order", kind="stable").drop(columns=["order"]).reset_index(drop=True)

TABLE_ID = "pf-table"
KEY_SHOW_MORE = "pf_show_more"
if KEY_SHOW_MORE not in st.session_state:
    st.session_state[KEY_SHOW_MORE] = False

# Titre « Holdings (current) » + bouton « Voir plus » aligné, un peu à droite
c_title, c_btn, c_spacer = st.columns([0.25, 0.14, 0.61])  # ajuste ces ratios si besoin
with c_title:
    st.markdown("### Holdings (current)")
with c_btn:
    # petit offset vertical pour aligner avec la baseline du H3
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.checkbox("See all", key=KEY_SHOW_MORE)


show_more = st.session_state[KEY_SHOW_MORE]

# CSS: colonnes auto (table-layout:auto), zébrage, sticky header, masquage des lignes >=6 si !show_more
css = f"""
<style>
#{TABLE_ID} {{
  width: 100%;
  border-collapse: collapse;
  table-layout: auto;
}}
#{TABLE_ID} thead th {{
  position: sticky; top: 0; z-index: 1;
  background: var(--background-color, #111);
}}
#{TABLE_ID} th, #{TABLE_ID} td {{
  padding: 6px 10px;
  border-bottom: 1px solid rgba(128,128,128,0.25);
  text-align: right;
  vertical-align: top;
  white-space: normal;
  word-break: normal;
}}
#{TABLE_ID} th:first-child, #{TABLE_ID} td:first-child {{ text-align: left; }}

/* Zébrage */
@media (prefers-color-scheme: dark) {{
  #{TABLE_ID} tbody tr:nth-child(even) {{ background-color: rgba(255,255,255,0.06); }}
}}
@media (prefers-color-scheme: light) {{
  #{TABLE_ID} tbody tr:nth-child(even) {{ background-color: rgba(0,0,0,0.04); }}
}}

/* Masquer les lignes à partir de la 6e en mode aperçu */
{f"#{TABLE_ID} tbody tr:nth-child(n+5) {{ display: none; }}" if not show_more else ""}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

html_full = view.to_html(index=False, escape=False, classes="portfolio", table_id=TABLE_ID)
st.markdown(html_full, unsafe_allow_html=True)

# -------- courbe NAV (Altair) : survol continu affiche la valeur au x courant --------
st.markdown("### Portfolio value over time")
nav_ts = (nav_incl_cash if include_cash_ts else nav_ex_cash)
if nav_ts.empty:
    st.info("No time series to plot for the selected range.")
else:
    y_min = float(nav_ts.min())
    y_max = float(nav_ts.max())
    pad = 0.005 * (y_max - y_min) if y_max > y_min else 1.0
    domain = [y_min - pad, y_max + pad]

    chart_df = nav_ts.reset_index()
    chart_df.columns = ["Date", "Portfolio (CAD)"]
    # bandes transparentes par intervalle [Date_i, Date_{i+1}) pour capturer la souris partout
    chart_df["Date_next"] = chart_df["Date"].shift(-1)
    # pour la dernière, étendre d'un jour
    chart_df["Date_next"] = chart_df["Date_next"].fillna(chart_df["Date"] + pd.Timedelta(days=1))

    # sélection point le long de l'axe x, mise à jour en continu
    hover = alt.selection_point(fields=["Date"], on="mousemove", nearest=True, empty=False)

    base = alt.Chart(chart_df).properties(height=380)

    # couche de capture (rectangles transparents couvrant tout l'intervalle en x)
    capture = (
        base.mark_rect(opacity=0)
        .encode(x="Date:T", x2="Date_next:T")
        .add_params(hover)
    )

    line = base.mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Portfolio (CAD):Q", title="Value (CAD)", scale=alt.Scale(domain=domain)),
    )

    rule = base.mark_rule().encode(
        x="Date:T",
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Portfolio (CAD):Q", format=",.2f")],
    ).transform_filter(hover)

    point = base.mark_circle(size=45).encode(
        x="Date:T",
        y="Portfolio (CAD):Q",
    ).transform_filter(hover)

    label = base.mark_text(align="left", dx=6, dy=-8).encode(
        x="Date:T",
        y="Portfolio (CAD):Q",
        text=alt.Text("Portfolio (CAD):Q", format=",.2f"),
    ).transform_filter(hover)

    chart = alt.layer(line, capture, rule, point, label).interactive()

    st.altair_chart(chart, use_container_width=True)

# -------- téléchargements --------
with st.expander("Download data"):
    export_df = positions.rename(columns={
        "ticker": "Ticker",
        "quantity": "Quantity",
        "avg_cost": "Avg Cost",
    })[[
        "Ticker", "Quantity", "Price", "Avg Cost",
        "Mkt Value", "Book Cost", "Gain/Loss Unrealized",
        "% of Portfolio", "Start Price", "End Price",
        "Return % (Window)", "Return % (Since buy)",
    ]]
    st.download_button(
        "Download current holdings (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="current_holdings.csv",
        mime="text/csv",
    )

    nav_ts_export = (nav_incl_cash if include_cash_ts else nav_ex_cash).rename("Portfolio (CAD)")
    st.download_button(
        "Download portfolio timeseries (CSV)",
        data=nav_ts_export.to_frame().to_csv(index=True).encode("utf-8"),
        file_name="portfolio_timeseries.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Survol continu: la règle et la valeur suivent la position x de la souris dans tout le graphique.\n"
    "\nReturn % (Since buy) = Price/AvgCost - 1. Return % (Window) = EndPrice/StartPrice - 1.\n"
    "\nOrdre du tableau = ordre d'origine. Colonnes auto-ajustées; zébrage; « Voir plus » à droite du titre.\n"
    "\nLa série affichée inclut le cash par défaut."
)
