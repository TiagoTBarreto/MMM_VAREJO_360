
import pickle
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="MMM Budget Simulator", layout="wide")

logo_svg = """
<svg width="700" height="260" viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="260" rx="36" fill="#0F172A"/>
  <circle cx="105" cy="92" r="44" fill="#38BDF8"/>
  <circle cx="155" cy="135" r="44" fill="#22C55E"/>
  <circle cx="82" cy="157" r="34" fill="#F97316"/>
  <text x="230" y="105" font-family="Arial" font-size="54" font-weight="700" fill="white">Varejo</text>
  <text x="230" y="165" font-family="Arial" font-size="64" font-weight="800" fill="#38BDF8">360</text>
  <text x="232" y="205" font-family="Arial" font-size="22" fill="#CBD5E1">MMM Budget Simulator</text>
</svg>
"""
logo_b64 = base64.b64encode(logo_svg.encode("utf-8")).decode("utf-8")
st.sidebar.markdown(
    f'<img src="data:image/svg+xml;base64,{logo_b64}" style="width:100%; border-radius:18px; margin-bottom:16px;">',
    unsafe_allow_html=True
)

with open("mmm_artifacts.pkl", "rb") as f:
    artifact = pickle.load(f)

df_mmm         = artifact["df_mmm"]
media_cols     = artifact["media_cols"]
adstock_params = artifact["adstock_params"]
hill_params    = artifact["hill_params"]
all_results    = artifact["all_results"]
X_columns      = artifact["X_columns"]

df_mmm["date"] = pd.to_datetime(df_mmm["date"])
df_mmm = df_mmm.sort_values("date").reset_index(drop=True)

historical_adstock = {}

for ch in media_cols:
    alpha = adstock_params[ch]
    spend = df_mmm[ch].values

    ah = np.zeros(len(spend))
    ah[0] = spend[0]

    for t in range(1, len(spend)):
        ah[t] = spend[t] + alpha * ah[t - 1]

    historical_adstock[ch] = ah

initial_adstock = {ch: historical_adstock[ch][-1] for ch in media_cols}

def hill_function(x, k, s):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    return (x**s) / ((x**s) + (k**s) + 1e-9)

def get_hill_params(channel):
    value = hill_params[channel]
    adstock_hist = historical_adstock[channel]
    valid_adstock = adstock_hist[adstock_hist > 0]

    default_k = (
        np.median(valid_adstock)
        if len(valid_adstock) > 0
        else np.median(df_mmm[channel].values)
    )

    if isinstance(value, dict):
        return value.get("k", default_k), value.get("s", 2.0)

    return default_k, float(value)

def get_beta(channel, coef_original):
    idx = X_columns.index(f"{channel}_hill")
    return coef_original[idx]

def response_from_adstock(adstock_series, channel, beta):
    k, s = get_hill_params(channel)
    return beta * hill_function(adstock_series, k=k, s=s)

def simulate_adstock_series(spend_series_or_scalar, channel, n_weeks):
    alpha = adstock_params[channel]

    if np.isscalar(spend_series_or_scalar):
        spend_arr = np.full(n_weeks, spend_series_or_scalar)
    else:
        spend_arr = np.asarray(spend_series_or_scalar, dtype=float)

    ads = np.zeros(n_weeks)
    ads[0] = spend_arr[0] + alpha * initial_adstock[channel]

    for t in range(1, n_weeks):
        ads[t] = spend_arr[t] + alpha * ads[t - 1]

    return ads

def simulate_adstock_from_initial(spend_arr, channel, initial_value):
    alpha = adstock_params[channel]
    spend_arr = np.asarray(spend_arr, dtype=float)

    ads = np.zeros(len(spend_arr))
    ads[0] = spend_arr[0] + alpha * initial_value

    for t in range(1, len(spend_arr)):
        ads[t] = spend_arr[t] + alpha * ads[t - 1]

    return ads

def split_media_effect(spend_arr, channel, beta, n_weeks):
    ads_total = simulate_adstock_series(spend_arr, channel, n_weeks)
    ads_carryover_only = simulate_adstock_series(np.zeros(n_weeks), channel, n_weeks)

    total_effect = response_from_adstock(ads_total, channel, beta)
    carryover_effect = response_from_adstock(ads_carryover_only, channel, beta)
    new_investment_effect = total_effect - carryover_effect

    return total_effect, carryover_effect, new_investment_effect, ads_total, ads_carryover_only

def new_investment_response_from_spend(spend_arr, channel, beta, n_weeks):
    _, _, new_effect, _, _ = split_media_effect(spend_arr, channel, beta, n_weeks)
    return new_effect

def saturation_pct(adstock_series, channel):
    k, s = get_hill_params(channel)
    return hill_function(adstock_series, k=k, s=s) * 100

def clean(ch):
    return (
        ch.replace("_spend_brl", "")
        .replace("_hill", "")
        .replace("_", " ")
        .title()
    )

def optimize_2d_raw(budget_total, coef_original, n_weeks, step_size):
    spend = {ch: np.zeros(n_weeks) for ch in media_cols}
    remaining = budget_total

    while remaining > 1e-6:
        step = min(step_size, remaining)
        best = (None, None)
        best_gain = -np.inf

        for ch in media_cols:
            beta = get_beta(ch, coef_original)

            cur_resp = new_investment_response_from_spend(
                spend[ch], ch, beta, n_weeks
            ).sum()

            for w in range(n_weeks):
                new_spend = spend[ch].copy()
                new_spend[w] += step

                new_resp = new_investment_response_from_spend(
                    new_spend, ch, beta, n_weeks
                ).sum()

                gain = new_resp - cur_resp

                if gain > best_gain:
                    best_gain = gain
                    best = (ch, w)

        ch_best, w_best = best

        if ch_best is None:
            break

        spend[ch_best][w_best] += step
        remaining -= step

    return spend

@st.cache_data(show_spinner=False)
def cached_optimize_2d(budget_total, model_name, step_size, n_weeks, coef_tuple):
    coef_original = np.array(coef_tuple, dtype=float)
    return optimize_2d_raw(budget_total, coef_original, n_weeks, step_size)

@st.cache_data(show_spinner=False)
def cached_prophet_baseline(model_name, coef_tuple, weeks_tuple):
    coef_original = np.array(coef_tuple, dtype=float)
    weeks = pd.to_datetime(list(weeks_tuple))

    prophet_df = df_mmm[["date", "revenue_brl"]].copy()
    prophet_df = prophet_df.rename(columns={"date": "ds", "revenue_brl": "y"})

    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    prophet_model.fit(prophet_df)

    future_df = pd.DataFrame({"ds": weeks})
    forecast_future = prophet_model.predict(future_df)

    future_controls = pd.DataFrame({
        "date": weeks,
        "trend": forecast_future["trend"].values,
        "yearly": forecast_future["yearly"].values,
    })

    hist_controls = df_mmm.copy()
    hist_controls["weekofyear"] = hist_controls["date"].dt.isocalendar().week.astype(int)
    future_controls["weekofyear"] = future_controls["date"].dt.isocalendar().week.astype(int)

    future_controls["economic_confidence_index"] = (
        hist_controls["economic_confidence_index"].tail(8).mean()
        if "economic_confidence_index" in hist_controls.columns
        else 0
    )

    if "competitor_promo_index" in hist_controls.columns:
        promo_by_week = hist_controls.groupby("weekofyear")["competitor_promo_index"].mean()
        future_controls["competitor_promo_index"] = (
            future_controls["weekofyear"]
            .map(promo_by_week)
            .fillna(hist_controls["competitor_promo_index"].mean())
        )
    else:
        future_controls["competitor_promo_index"] = 0

    if "is_black_friday" in hist_controls.columns:
        bf_by_week = hist_controls.groupby("weekofyear")["is_black_friday"].max()
        future_controls["is_black_friday"] = (
            future_controls["weekofyear"]
            .map(bf_by_week)
            .fillna(0)
            .astype(int)
        )
    else:
        future_controls["is_black_friday"] = 0

    if "holiday_week" in hist_controls.columns:
        holiday_by_week = hist_controls.groupby("weekofyear")["holiday_week"].max()
        future_controls["holiday_week"] = (
            future_controls["weekofyear"]
            .map(holiday_by_week)
            .fillna(0)
            .astype(int)
        )
    else:
        future_controls["holiday_week"] = 0

    res = all_results[model_name]

    intercept = (
        res["intercept_original"]
        if "intercept_original" in res
        else res["intercept"]
        if "intercept" in res
        else 0
    )

    baseline_weekly = np.full(len(weeks), intercept, dtype=float)

    baseline_features = [
        "trend",
        "yearly",
        "competitor_promo_index",
        "economic_confidence_index",
        "is_black_friday",
        "holiday_week"
    ]

    for feature in baseline_features:
        if feature in X_columns:
            idx = X_columns.index(feature)
            coef = coef_original[idx]
            baseline_weekly += future_controls[feature].values * coef

    baseline_weekly = np.maximum(baseline_weekly, 0)

    return baseline_weekly, future_controls

def normalized_weights_to_budget(weights_dict, total_budget):
    weights = pd.Series(weights_dict, dtype=float).clip(lower=0)

    if weights.sum() <= 0:
        weights[:] = 1

    shares = weights / weights.sum()
    budget = shares * total_budget

    return shares.to_dict(), budget.to_dict()

def build_historical_q1_analysis(coef_original, weekly_spend_plan, simulation):
    hist = df_mmm.copy().reset_index(drop=True)
    hist["_row_id"] = np.arange(len(hist))
    hist["year"] = hist["date"].dt.year
    hist["week"] = hist["date"].dt.isocalendar().week.astype(int)

    raw_media_cols = [
        c for c in hist.columns
        if "spend" in c.lower()
        and "adstock" not in c.lower()
        and "total" not in c.lower()
        and "hill" not in c.lower()
    ]

    hist["total_media_spend"] = hist[raw_media_cols].sum(axis=1)

    first_13 = hist[hist["week"] <= 13].copy()

    year_rows = []
    channel_rows = []

    for year, g in first_13.groupby("year"):
        g = g.sort_values("date").copy()
        row_ids = g["_row_id"].values

        revenue_year = g["revenue_brl"].sum()
        spend_year = g[raw_media_cols].sum().sum()

        total_media_effect_year = 0
        carryover_effect_year = 0
        period_investment_effect_year = 0

        for ch in media_cols:
            beta = get_beta(ch, coef_original)
            alpha = adstock_params[ch]
            cn = ch.replace("_spend_brl", "")

            spend_arr = g[ch].values
            spend_ch = spend_arr.sum()

            total_adstock_ch = historical_adstock[ch][row_ids]

            first_idx = row_ids[0]
            initial_before_period = historical_adstock[ch][first_idx - 1] if first_idx > 0 else 0

            carryover_adstock_ch = simulate_adstock_from_initial(
                np.zeros(len(g)),
                ch,
                initial_before_period
            )

            total_effect_ch = response_from_adstock(total_adstock_ch, ch, beta)
            carryover_effect_ch = response_from_adstock(carryover_adstock_ch, ch, beta)
            period_effect_ch = total_effect_ch - carryover_effect_ch

            total_media_effect_year += total_effect_ch.sum()
            carryover_effect_year += carryover_effect_ch.sum()
            period_investment_effect_year += period_effect_ch.sum()

            channel_rows.append({
                "year": year,
                "Canal": clean(ch),
                "Spend (R$)": spend_ch,
                "Incremental Total com Carryover (R$)": total_effect_ch.sum(),
                "Incremental Carryover Pré-Período (R$)": carryover_effect_ch.sum(),
                "Incremental Só Investimento 13 Semanas (R$)": period_effect_ch.sum(),
                "ROI com Carryover": total_effect_ch.sum() / spend_ch if spend_ch > 0 else 0,
                "ROI Só Investimento 13 Semanas": period_effect_ch.sum() / spend_ch if spend_ch > 0 else 0,
                "Scenario": "Histórico"
            })

        year_rows.append({
            "year": year,
            "revenue_brl": revenue_year,
            "total_media_spend": spend_year,
            "Incremental Total com Carryover (R$)": total_media_effect_year,
            "Incremental Carryover Pré-Período (R$)": carryover_effect_year,
            "Incremental Só Investimento 13 Semanas (R$)": period_investment_effect_year,
            "ROI com Carryover": total_media_effect_year / spend_year if spend_year > 0 else 0,
            "ROI Só Investimento 13 Semanas": period_investment_effect_year / spend_year if spend_year > 0 else 0,
            "Scenario": "Histórico"
        })

    current_revenue = simulation["projected_revenue"].sum()
    current_spend = sum(weekly_spend_plan[ch].sum() for ch in media_cols)
    current_total_effect = simulation["total_media_effect"].sum()
    current_carryover_effect = simulation["carryover_media_effect"].sum()
    current_new_effect = simulation["new_media_effect"].sum()

    year_rows.append({
        "year": 2025,
        "revenue_brl": current_revenue,
        "total_media_spend": current_spend,
        "Incremental Total com Carryover (R$)": current_total_effect,
        "Incremental Carryover Pré-Período (R$)": current_carryover_effect,
        "Incremental Só Investimento 13 Semanas (R$)": current_new_effect,
        "ROI com Carryover": current_total_effect / current_spend if current_spend > 0 else 0,
        "ROI Só Investimento 13 Semanas": current_new_effect / current_spend if current_spend > 0 else 0,
        "Scenario": "Simulado 2025"
    })

    for ch in media_cols:
        cn = ch.replace("_spend_brl", "")
        spend_ch = weekly_spend_plan[ch].sum()
        total_effect_ch = simulation[f"{cn}_total_media_effect"].sum()
        carryover_effect_ch = simulation[f"{cn}_carryover_effect"].sum()
        new_effect_ch = simulation[f"{cn}_new_investment_effect"].sum()

        channel_rows.append({
            "year": 2025,
            "Canal": clean(ch),
            "Spend (R$)": spend_ch,
            "Incremental Total com Carryover (R$)": total_effect_ch,
            "Incremental Carryover Pré-Período (R$)": carryover_effect_ch,
            "Incremental Só Investimento 13 Semanas (R$)": new_effect_ch,
            "ROI com Carryover": total_effect_ch / spend_ch if spend_ch > 0 else 0,
            "ROI Só Investimento 13 Semanas": new_effect_ch / spend_ch if spend_ch > 0 else 0,
            "Scenario": "Simulado 2025"
        })

    year_summary = pd.DataFrame(year_rows).sort_values("year")
    channel_compare = pd.DataFrame(channel_rows)

    channel_compare["Spend Share (%)"] = (
        channel_compare["Spend (R$)"]
        / channel_compare.groupby("year")["Spend (R$)"].transform("sum")
        * 100
    ).fillna(0)

    return year_summary, channel_compare

st.sidebar.title("Inputs do Simulador")

model_name = st.sidebar.selectbox(
    "Modelo",
    list(all_results.keys()),
    index=list(all_results.keys()).index("Ridge") if "Ridge" in all_results else 0
)

res = all_results[model_name]
coef_original = res["coef_original"]
coef_tuple = tuple(np.asarray(coef_original, dtype=float).round(12))
roi_table = res["roi_table"]
contributions = res["contributions"].copy()
contributions["date"] = pd.to_datetime(contributions["date"])

budget_total = st.sidebar.number_input(
    "Budget total Q1 2025",
    min_value=0,
    value=3_000_000,
    step=100_000
)

allocation_mode = st.sidebar.radio(
    "Modo de alocação",
    ["Otimizado (2D)", "Manual"]
)

step_size = st.sidebar.number_input(
    "Granularidade otimização",
    min_value=1000,
    value=50000,
    step=5000
)

st.sidebar.caption(
    "A otimização fica em cache. Ela só recalcula quando os inputs da barra lateral mudam."
)

weeks = pd.date_range("2025-01-06", "2025-03-31", freq="W-MON")
n_weeks = len(weeks)
week_labels = [str(w.date()) for w in weeks]
weeks_tuple = tuple(weeks.astype(str))

baseline_weekly, future_controls = cached_prophet_baseline(
    model_name=model_name,
    coef_tuple=coef_tuple,
    weeks_tuple=weeks_tuple
)

st.title("MMM Interactive Budget Simulator")
st.caption("Simulador de budget, ROI, response curve e projeção de vendas - Q1 2025")

st.subheader("1. Alocação de budget por canal")

if allocation_mode == "Manual":
    st.markdown("**Distribuição por canal — use pesos. O app normaliza para 100%.**")

    cols = st.columns(3)
    channel_weights = {}

    for i, ch in enumerate(media_cols):
        with cols[i % 3]:
            channel_weights[ch] = st.slider(
                clean(ch),
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                key=f"channel_weight_{ch}"
            )

    channel_shares, channel_budget = normalized_weights_to_budget(
        channel_weights,
        budget_total
    )

    channel_share_df = pd.DataFrame({
        "Canal": [clean(ch) for ch in media_cols],
        "Peso": [channel_weights[ch] for ch in media_cols],
        "Share Normalizado (%)": [channel_shares[ch] * 100 for ch in media_cols],
        "Budget Canal (R$)": [channel_budget[ch] for ch in media_cols]
    }).sort_values("Share Normalizado (%)", ascending=False)

    st.dataframe(
        channel_share_df.style.format({
            "Share Normalizado (%)": "{:.1f}%",
            "Budget Canal (R$)": "R$ {:,.0f}"
        }),
        use_container_width=True
    )

    st.markdown("**Distribuição semanal — use pesos. O app normaliza cada canal para 100%.**")

    weekly_spend_plan = {}

    for ch in media_cols:
        with st.expander(f"{clean(ch)} — budget total: R$ {channel_budget[ch]:,.0f}", expanded=False):
            wcols = st.columns(4)
            week_weights = {}

            for w in range(n_weeks):
                with wcols[w % 4]:
                    week_weights[w] = st.slider(
                        week_labels[w],
                        min_value=0,
                        max_value=100,
                        value=50,
                        step=1,
                        key=f"week_weight_{ch}_{w}"
                    )

            week_weights_series = pd.Series(week_weights, dtype=float).clip(lower=0)

            if week_weights_series.sum() <= 0:
                week_weights_series[:] = 1

            weekly_shares = week_weights_series / week_weights_series.sum()
            weekly_spend_plan[ch] = weekly_shares.values * channel_budget[ch]

            weekly_preview = pd.DataFrame({
                "Semana": week_labels,
                "Peso": week_weights_series.values,
                "Share Normalizado (%)": weekly_shares.values * 100,
                "Spend (R$)": weekly_spend_plan[ch]
            })

            st.dataframe(
                weekly_preview.style.format({
                    "Share Normalizado (%)": "{:.1f}%",
                    "Spend (R$)": "R$ {:,.0f}"
                }),
                use_container_width=True
            )

else:
    with st.spinner("Optimizing 2D allocation channel x week..."):
        weekly_spend_plan = cached_optimize_2d(
            budget_total=budget_total,
            model_name=model_name,
            step_size=step_size,
            n_weeks=n_weeks,
            coef_tuple=coef_tuple
        )

alloc_rows = []

for ch in media_cols:
    total_ch = weekly_spend_plan[ch].sum()
    alloc_rows.append({
        "Canal": clean(ch),
        "Q1 Total (R$)": total_ch,
        "Share (%)": 0
    })

alloc_df = pd.DataFrame(alloc_rows)

if alloc_df["Q1 Total (R$)"].sum() > 0:
    alloc_df["Share (%)"] = alloc_df["Q1 Total (R$)"] / alloc_df["Q1 Total (R$)"].sum() * 100
else:
    alloc_df["Share (%)"] = 0

alloc_df = alloc_df.sort_values("Q1 Total (R$)", ascending=False).reset_index(drop=True)

fig_alloc = px.bar(
    alloc_df,
    x="Canal",
    y="Q1 Total (R$)",
    color="Canal",
    title="BUDGET Q1 POR CANAL",
    text_auto=".2s",
    category_orders={"Canal": alloc_df["Canal"].tolist()}
)

# Cumulative investment line
alloc_df["Investimento Acumulado (R$)"] = alloc_df["Q1 Total (R$)"].cumsum()

fig_alloc.add_scatter(
    x=alloc_df["Canal"],
    y=alloc_df["Investimento Acumulado (R$)"],
    mode="lines+markers+text",
    name="Investimento acumulado",
    text=[f"R$ {v:,.0f}" for v in alloc_df["Investimento Acumulado (R$)"]],
    textposition="top center",
    line=dict(width=3)
)

# Channel share as markers only, using the same money axis converted to BRL scale
total_budget_alloc = alloc_df["Q1 Total (R$)"].sum()

alloc_df["Share no eixo R$"] = (
    alloc_df["Share (%)"] / 100 * total_budget_alloc
)

fig_alloc.add_scatter(
    x=alloc_df["Canal"],
    y=alloc_df["Share no eixo R$"],
    mode="markers+text",
    name="% do canal",
    text=[f"{v:.1f}%" for v in alloc_df["Share (%)"]],
    textposition="bottom center",
    marker=dict(size=12)
)

fig_alloc.update_yaxes(
    title="Investimento Q1 / Acumulado (R$)",
    tickformat=",.0f"
)

fig_alloc.update_layout(
    barmode="group",
    legend_title_text="Métrica"
)

st.plotly_chart(fig_alloc, use_container_width=True)

spend_matrix = pd.DataFrame(
    {clean(ch): weekly_spend_plan[ch] for ch in media_cols},
    index=week_labels
)

spend_matrix = spend_matrix[alloc_df["Canal"].tolist()]

fig_heat = px.imshow(
    spend_matrix.T,
    text_auto=".2s",
    aspect="auto",
    title="Investimento semanal por canal (R$)",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_heat, use_container_width=True)

simulation = pd.DataFrame({"date": weeks})
simulation["baseline"] = baseline_weekly

total_media_effect = np.zeros(n_weeks)
carryover_media_effect = np.zeros(n_weeks)
new_media_effect = np.zeros(n_weeks)

for ch in media_cols:
    beta = get_beta(ch, coef_original)
    spend_arr = weekly_spend_plan[ch]

    total_effect, carryover_effect, new_effect, ads_total, ads_carryover_only = split_media_effect(
        spend_arr=spend_arr,
        channel=ch,
        beta=beta,
        n_weeks=n_weeks
    )

    sat_total = saturation_pct(ads_total, ch)
    sat_carryover = saturation_pct(ads_carryover_only, ch)

    cn = ch.replace("_spend_brl", "")

    simulation[f"{cn}_spend"] = spend_arr
    simulation[f"{cn}_adstock_total"] = ads_total
    simulation[f"{cn}_adstock_carryover_only"] = ads_carryover_only
    simulation[f"{cn}_total_media_effect"] = total_effect
    simulation[f"{cn}_carryover_effect"] = carryover_effect
    simulation[f"{cn}_new_investment_effect"] = new_effect
    simulation[f"{cn}_saturation_total"] = sat_total
    simulation[f"{cn}_saturation_carryover_only"] = sat_carryover

    total_media_effect += total_effect
    carryover_media_effect += carryover_effect
    new_media_effect += new_effect

simulation["carryover_media_effect"] = carryover_media_effect
simulation["new_media_effect"] = new_media_effect
simulation["total_media_effect"] = total_media_effect

simulation["projected_revenue"] = (
    simulation["baseline"]
    + simulation["carryover_media_effect"]
    + simulation["new_media_effect"]
)

st.subheader("2. Resultado projetado - Q1 2025")

total_revenue = simulation["projected_revenue"].sum()
total_baseline = simulation["baseline"].sum()
total_carryover = simulation["carryover_media_effect"].sum()
total_new_media = simulation["new_media_effect"].sum()
total_media = simulation["total_media_effect"].sum()

projected_roi_new_budget = total_new_media / budget_total if budget_total > 0 else 0
projected_roi_total_media = total_media / budget_total if budget_total > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Budget Total 2025", f"R$ {budget_total:,.0f}")
k2.metric("Baseline Projetado", f"R$ {total_baseline:,.0f}")
k3.metric("Carryover 2024", f"R$ {total_carryover:,.0f}")
k4.metric("Incremental Budget 2025", f"R$ {total_new_media:,.0f}")
k5.metric("Venda Total Projetada", f"R$ {total_revenue:,.0f}")

r1, r2 = st.columns(2)
r1.metric("ROI Só Investimento 2025", f"{projected_roi_new_budget:.2f}x")
r2.metric("ROI com Carryover", f"{projected_roi_total_media:.2f}x")

st.subheader("2.1 ROI 2025 por canal")

roi_sim_rows = []

for ch in media_cols:
    spend_arr = weekly_spend_plan[ch]
    cn = ch.replace("_spend_brl", "")

    total_spend = spend_arr.sum()
    carry_resp = simulation[f"{cn}_carryover_effect"].sum()
    new_resp = simulation[f"{cn}_new_investment_effect"].sum()
    total_resp = simulation[f"{cn}_total_media_effect"].sum()

    roi_sim_rows.append({
        "Canal": clean(ch),
        "Investimento 2025 (R$)": total_spend,
        "Incremental Total com Carryover (R$)": total_resp,
        "Incremental Carryover 2024 (R$)": carry_resp,
        "Incremental Só Investimento 2025 (R$)": new_resp,
        "ROI com Carryover": total_resp / total_spend if total_spend > 0 else 0,
        "ROI Só Investimento 2025": new_resp / total_spend if total_spend > 0 else 0
    })

roi_sim_df = pd.DataFrame(roi_sim_rows).sort_values(
    "ROI Só Investimento 2025",
    ascending=False
)

st.dataframe(
    roi_sim_df.style.format({
        "Investimento 2025 (R$)": "R$ {:,.0f}",
        "Incremental Total com Carryover (R$)": "R$ {:,.0f}",
        "Incremental Carryover 2024 (R$)": "R$ {:,.0f}",
        "Incremental Só Investimento 2025 (R$)": "R$ {:,.0f}",
        "ROI com Carryover": "{:.2f}x",
        "ROI Só Investimento 2025": "{:.2f}x"
    }),
    use_container_width=True
)

fig_roi_compare_2025 = px.bar(
    roi_sim_df.melt(
        id_vars="Canal",
        value_vars=["ROI com Carryover", "ROI Só Investimento 2025"],
        var_name="Visão",
        value_name="ROI"
    ),
    x="Canal",
    y="ROI",
    color="Visão",
    barmode="group",
    title="ROI 2025 por canal: com carryover vs só investimento do período"
)

fig_roi_compare_2025.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Break-even")
st.plotly_chart(fig_roi_compare_2025, use_container_width=True)

st.subheader("3. Projeção semanal de vendas")

decomp_df = simulation[[
    "date",
    "baseline",
    "carryover_media_effect",
    "new_media_effect",
    "projected_revenue"
]].copy()

decomp_long = decomp_df.melt(
    id_vars="date",
    value_vars=["baseline", "carryover_media_effect", "new_media_effect", "projected_revenue"],
    var_name="Metrica",
    value_name="Valor"
)

decomp_long["Metrica"] = decomp_long["Metrica"].map({
    "baseline": "Baseline Prophet + MMM Controls",
    "carryover_media_effect": "Carryover de Investimentos Anteriores",
    "new_media_effect": "Incremental Só Investimento do Período",
    "projected_revenue": "Venda Projetada"
})

fig_sales = px.line(
    decomp_long,
    x="date",
    y="Valor",
    color="Metrica",
    markers=True,
    title="Predicted Revenue Decomposition"
)

fig_sales.update_yaxes(tickformat=",.0f", title="Predicted Revenue BRL")
fig_sales.update_xaxes(title="Date")
st.plotly_chart(fig_sales, use_container_width=True)

st.subheader("3.1 Decomposição empilhada")

stack_long = decomp_df.melt(
    id_vars="date",
    value_vars=["baseline", "carryover_media_effect", "new_media_effect"],
    var_name="Componente",
    value_name="Valor"
)

stack_long["Componente"] = stack_long["Componente"].map({
    "baseline": "Baseline",
    "carryover_media_effect": "Carryover Anterior",
    "new_media_effect": "Incremental Investimento do Período"
})

fig_stack = px.area(
    stack_long,
    x="date",
    y="Valor",
    color="Componente",
    title="Revenue Decomposition: Baseline vs Carryover vs New Budget"
)

fig_stack.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig_stack, use_container_width=True)

st.subheader("3.2 Contribuição semanal de mídia por canal")

total_effect_cols = [f"{ch.replace('_spend_brl', '')}_total_media_effect" for ch in media_cols]
carryover_effect_cols = [f"{ch.replace('_spend_brl', '')}_carryover_effect" for ch in media_cols]
new_effect_cols = [f"{ch.replace('_spend_brl', '')}_new_investment_effect" for ch in media_cols]

total_media_long = simulation.melt(
    id_vars="date",
    value_vars=total_effect_cols,
    var_name="Canal",
    value_name="Efeito Total Mídia (R$)"
)

total_media_long["Canal"] = (
    total_media_long["Canal"]
    .str.replace("_total_media_effect", "", regex=False)
    .str.replace("_", " ", regex=False)
    .str.title()
)

fig_total_media_weekly = px.area(
    total_media_long,
    x="date",
    y="Efeito Total Mídia (R$)",
    color="Canal",
    title="Weekly Total Media Effect by Channel"
)

fig_total_media_weekly.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig_total_media_weekly, use_container_width=True)

tab_carry, tab_new = st.tabs(["Carryover anterior por canal", "Incremental investimento do período por canal"])

with tab_carry:
    carryover_long = simulation.melt(
        id_vars="date",
        value_vars=carryover_effect_cols,
        var_name="Canal",
        value_name="Carryover Anterior (R$)"
    )

    carryover_long["Canal"] = (
        carryover_long["Canal"]
        .str.replace("_carryover_effect", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    fig_carry = px.area(
        carryover_long,
        x="date",
        y="Carryover Anterior (R$)",
        color="Canal",
        title="Weekly Carryover Effect from Previous Investments"
    )
    fig_carry.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig_carry, use_container_width=True)

with tab_new:
    new_long = simulation.melt(
        id_vars="date",
        value_vars=new_effect_cols,
        var_name="Canal",
        value_name="Incremental Investimento do Período (R$)"
    )

    new_long["Canal"] = (
        new_long["Canal"]
        .str.replace("_new_investment_effect", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    fig_new = px.area(
        new_long,
        x="date",
        y="Incremental Investimento do Período (R$)",
        color="Canal",
        title="Weekly Incremental Effect from Period Budget"
    )
    fig_new.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig_new, use_container_width=True)

st.subheader("4. Adstock efetivo por canal ao longo do Q1")

adstock_total_cols = [c for c in simulation.columns if c.endswith("_adstock_total")]

ads_long = simulation.melt(
    id_vars="date",
    value_vars=adstock_total_cols,
    var_name="Canal",
    value_name="Adstock Total"
)

ads_long["Canal"] = (
    ads_long["Canal"]
    .str.replace("_adstock_total", "", regex=False)
    .str.replace("_", " ", regex=False)
    .str.title()
)

fig_ads = px.line(
    ads_long,
    x="date",
    y="Adstock Total",
    color="Canal",
    markers=True,
    title="Adstock Total: Carryover + Spend do Período"
)

fig_ads.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig_ads, use_container_width=True)

st.subheader("4.1 Saturação semanal por canal (%)")

sat_total_cols = [c for c in simulation.columns if c.endswith("_saturation_total")]

sat_long = simulation.melt(
    id_vars="date",
    value_vars=sat_total_cols,
    var_name="Canal",
    value_name="Saturacao Total (%)"
)

sat_long["Canal"] = (
    sat_long["Canal"]
    .str.replace("_saturation_total", "", regex=False)
    .str.replace("_", " ", regex=False)
    .str.title()
)



fig_sat = px.line(
    sat_long,
    x="date",
    y="Saturacao Total (%)",
    color="Canal",
    markers=True,
    title="Saturação semanal total por canal (%)",
    range_y=[0, 100]
)

fig_sat.update_traces(
    mode="lines+markers+text",
    texttemplate="%{y:.1f}%",
    textposition="top center"
)

fig_sat.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Alta saturação 80%")
fig_sat.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Saturação média 50%")
st.plotly_chart(fig_sat, use_container_width=True)

st.subheader("5. Plano semanal sugerido")

spend_cols = [c for c in simulation.columns if c.endswith("_spend")]

display_cols = [
    "date",
    "baseline",
    "carryover_media_effect",
    "new_media_effect",
    "total_media_effect",
    "projected_revenue"
] + spend_cols

weekly_plan = simulation[display_cols].copy()

st.dataframe(
    weekly_plan.style.format({
        col: "R$ {:,.0f}"
        for col in weekly_plan.columns
        if col != "date"
    }),
    use_container_width=True
)

st.subheader("6. Análise histórica das primeiras 13 semanas")

hist_summary_plus, hist_channel_compare = build_historical_q1_analysis(
    coef_original=coef_original,
    weekly_spend_plan=weekly_spend_plan,
    simulation=simulation
)

col_h1, col_h2 = st.columns(2)

with col_h1:
    hist_effect_plot = hist_summary_plus.melt(
        id_vars="year",
        value_vars=[
            "total_media_spend",
            "Incremental Total com Carryover (R$)",
            "Incremental Só Investimento 13 Semanas (R$)"
        ],
        var_name="Métrica",
        value_name="Valor"
    )

    fig_hist_effect = px.bar(
        hist_effect_plot,
        x="year",
        y="Valor",
        color="Métrica",
        barmode="group",
        title="Q1 semanas 1-13: Spend vs Incremental com/sem carryover"
    )

    fig_hist_effect.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig_hist_effect, use_container_width=True)

with col_h2:
    hist_roi_plot = hist_summary_plus.melt(
        id_vars="year",
        value_vars=["ROI com Carryover", "ROI Só Investimento 13 Semanas"],
        var_name="Visão ROI",
        value_name="ROI"
    )

    fig_hist_roi = px.line(
        hist_roi_plot,
        x="year",
        y="ROI",
        color="Visão ROI",
        markers=True,
        title="ROI Ads: com carryover vs só investimento do período"
    )

    fig_hist_roi.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig_hist_roi, use_container_width=True)

st.markdown("**Spend por canal nas primeiras 13 semanas — histórico vs simulado**")

fig_hist_channel = px.bar(
    hist_channel_compare,
    x="year",
    y="Spend (R$)",
    color="Canal",
    text="Spend Share (%)",
    title="Spend por canal com share percentual dentro de cada ano",
    barmode="stack"
)

fig_hist_channel.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
fig_hist_channel.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig_hist_channel, use_container_width=True)

st.markdown("**ROI por canal — com carryover vs só investimento do período**")

roi_channel_long = hist_channel_compare.melt(
    id_vars=["year", "Canal", "Scenario"],
    value_vars=["ROI com Carryover", "ROI Só Investimento 13 Semanas"],
    var_name="Visão ROI",
    value_name="ROI"
)

fig_channel_roi = px.bar(
    roi_channel_long,
    x="Canal",
    y="ROI",
    color="Visão ROI",
    barmode="group",
    facet_col="year",
    facet_col_wrap=4,
    title="ROI Ads por canal nas primeiras 13 semanas"
)

fig_channel_roi.update_yaxes(tickformat=".2f")
st.plotly_chart(fig_channel_roi, use_container_width=True)

with st.expander("Tabela detalhada: histórico vs simulado"):
    st.dataframe(
        hist_channel_compare.sort_values(["year", "Spend (R$)"], ascending=[True, False]).style.format({
            "Spend (R$)": "R$ {:,.0f}",
            "Incremental Total com Carryover (R$)": "R$ {:,.0f}",
            "Incremental Carryover Pré-Período (R$)": "R$ {:,.0f}",
            "Incremental Só Investimento 13 Semanas (R$)": "R$ {:,.0f}",
            "ROI com Carryover": "{:.2f}x",
            "ROI Só Investimento 13 Semanas": "{:.2f}x",
            "Spend Share (%)": "{:.1f}%"
        }),
        use_container_width=True
    )

st.subheader("7. ROI histórico estimado pelo modelo")

st.dataframe(
    roi_table.style.format({
        "Incremental Revenue": "R$ {:,.0f}",
        "Spend": "R$ {:,.0f}",
        "Contribution (%)": "{:.1f}%",
        "ROI": "{:.2f}"
    }),
    use_container_width=True
)

st.subheader("8. Response curves por canal")

selected_channels = st.multiselect(
    "Canais para visualizar",
    media_cols,
    default=media_cols
)

curve_data = []

for ch in selected_channels:
    beta = get_beta(ch, coef_original)
    k, s = get_hill_params(ch)
    alpha = adstock_params[ch]
    carry = initial_adstock[ch]

    max_raw = max(
        df_mmm[ch].max() * 1.5,
        weekly_spend_plan[ch].max() * 2,
        np.median(historical_adstock[ch]) * 1.5,
        1
    )

    grid_spend = np.linspace(0, max_raw, 300)
    grid_ads_total = grid_spend + alpha * carry
    grid_ads_carryover = np.full_like(grid_ads_total, alpha * carry)

    resp_total = beta * hill_function(grid_ads_total, k=k, s=s)
    resp_carryover = beta * hill_function(grid_ads_carryover, k=k, s=s)
    resp_new = resp_total - resp_carryover
    sat_grid = hill_function(grid_ads_total, k=k, s=s) * 100

    curve_data.append(pd.DataFrame({
        "Adstock Efetivo (R$)": grid_ads_total,
        "Spend Novo (R$)": grid_spend,
        "Efeito Total Mídia (R$)": resp_total,
        "Carryover Anterior (R$)": resp_carryover,
        "Incremental Só Investimento do Período (R$)": resp_new,
        "Saturacao (%)": sat_grid,
        "Canal": clean(ch)
    }))

if curve_data:
    curve_df = pd.concat(curve_data)

    tab1, tab2, tab3 = st.tabs([
        "Incremental Investimento do Período",
        "Efeito Total Mídia",
        "Saturação"
    ])

    with tab1:
        fig_curve_new = px.line(
            curve_df,
            x="Spend Novo (R$)",
            y="Incremental Só Investimento do Período (R$)",
            color="Canal",
            title="Response Curve - Incremental do investimento do período vs Spend Novo"
        )
        fig_curve_new.update_yaxes(tickformat=",.0f")
        fig_curve_new.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig_curve_new, use_container_width=True)

    with tab2:
        fig_curve_total = px.line(
            curve_df,
            x="Adstock Efetivo (R$)",
            y="Efeito Total Mídia (R$)",
            color="Canal",
            title="Response Curve - Efeito Total de Mídia vs Adstock Efetivo"
        )
        fig_curve_total.update_yaxes(tickformat=",.0f")
        fig_curve_total.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig_curve_total, use_container_width=True)

    with tab3:
        fig_sat2 = px.line(
            curve_df,
            x="Adstock Efetivo (R$)",
            y="Saturacao (%)",
            color="Canal",
            title="Saturação vs Adstock efetivo",
            range_y=[0, 100]
        )
        fig_sat2.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Alta saturação")
        fig_sat2.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Saturação média")
        fig_sat2.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig_sat2, use_container_width=True)

with st.expander("Debug Prophet Baseline"):
    st.dataframe(
        future_controls.assign(baseline=baseline_weekly),
        use_container_width=True
    )

with st.expander("Debug Hill Params"):
    debug_rows = []

    for ch in media_cols:
        k, s = get_hill_params(ch)
        cn = ch.replace("_spend_brl", "")
        valid_hist = historical_adstock[ch][historical_adstock[ch] > 0]

        debug_rows.append({
            "Canal": clean(ch),
            "k": k,
            "s": s,
            "Historical adstock median": np.median(valid_hist) if len(valid_hist) > 0 else 0,
            "Q1 total adstock min": simulation[f"{cn}_adstock_total"].min(),
            "Q1 total adstock max": simulation[f"{cn}_adstock_total"].max(),
            "Q1 carryover adstock min": simulation[f"{cn}_adstock_carryover_only"].min(),
            "Q1 carryover adstock max": simulation[f"{cn}_adstock_carryover_only"].max(),
            "Q1 total saturation min": simulation[f"{cn}_saturation_total"].min(),
            "Q1 total saturation max": simulation[f"{cn}_saturation_total"].max()
        })

    debug_df = pd.DataFrame(debug_rows)

    st.dataframe(
        debug_df.style.format({
            "k": "{:,.2f}",
            "s": "{:.2f}",
            "Historical adstock median": "{:,.2f}",
            "Q1 total adstock min": "{:,.2f}",
            "Q1 total adstock max": "{:,.2f}",
            "Q1 carryover adstock min": "{:,.2f}",
            "Q1 carryover adstock max": "{:,.2f}",
            "Q1 total saturation min": "{:.2f}%",
            "Q1 total saturation max": "{:.2f}%"
        }),
        use_container_width=True
    )

st.subheader("9. Exportar simulação")

csv = weekly_plan.to_csv(index=False).encode("utf-8")

st.download_button(
    "Baixar plano semanal em CSV",
    csv,
    "mmm_q1_2025_weekly_plan.csv",
    "text/csv"
)
