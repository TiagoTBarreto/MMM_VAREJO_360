
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="MMM Budget Simulator", layout="wide")

with open("mmm_artifacts.pkl", "rb") as f:
    artifact = pickle.load(f)

df_mmm         = artifact["df_mmm"]
media_cols     = artifact["media_cols"]
adstock_params = artifact["adstock_params"]
hill_params    = artifact["hill_params"]
all_results    = artifact["all_results"]
X_columns      = artifact["X_columns"]

df_mmm["date"] = pd.to_datetime(df_mmm["date"])

# === HISTORICAL ADSTOCK ===
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

# === FUNCTIONS ===
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

def response_from_adstock(adstock_series, channel, beta):
    k, s = get_hill_params(channel)
    return beta * hill_function(adstock_series, k=k, s=s)

def split_media_effect(spend_arr, channel, beta, n_weeks):
    # Decomposes media effect into:
    # 1. carryover_effect: effect generated only by pre-2025 carryover
    # 2. new_investment_effect: incremental effect caused by 2025 planned spend
    # 3. total_effect: carryover + new investment effect
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

def optimize_2d(budget_total, coef_original, n_weeks, step_size=50000):
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

def forecast_baseline_with_prophet(weeks, coef_original, res):
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

    future_controls["competitor_promo_index"] = (
        df_mmm["competitor_promo_index"].median()
        if "competitor_promo_index" in df_mmm.columns
        else 0
    )

    future_controls["economic_confidence_index"] = (
        df_mmm["economic_confidence_index"].median()
        if "economic_confidence_index" in df_mmm.columns
        else 0
    )

    future_controls["is_black_friday"] = 0
    future_controls["holiday_week"] = 0

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

# === SIDEBAR ===
st.sidebar.title("Inputs do Simulador")

model_name = st.sidebar.selectbox(
    "Modelo",
    list(all_results.keys()),
    index=list(all_results.keys()).index("Ridge") if "Ridge" in all_results else 0
)

res = all_results[model_name]
coef_original = res["coef_original"]
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
    "Modo de alocacao",
    ["Otimizado (2D)", "Manual"]
)

step_size = st.sidebar.number_input(
    "Granularidade otimizacao",
    min_value=1000,
    value=50000,
    step=5000
)

st.sidebar.caption("Dica: granularidade maior deixa o app mais rápido. Ex: 50k ou 100k.")

# === DATE RANGE ===
weeks = pd.date_range("2025-01-06", "2025-03-31", freq="W-MON")
n_weeks = len(weeks)
week_labels = [str(w.date()) for w in weeks]

# === PROPHET BASELINE FORECAST ===
baseline_weekly, future_controls = forecast_baseline_with_prophet(
    weeks=weeks,
    coef_original=coef_original,
    res=res
)

st.title("MMM Interactive Budget Simulator")
st.caption("Simulador de budget, ROI, response curve e projeção de vendas - Q1 2025")

# === ALLOCATION ===
st.subheader("1. Alocação de budget por canal")

if allocation_mode == "Manual":
    st.markdown("**Distribuição por canal (%)**")

    cols = st.columns(3)
    manual_pct = {}

    for i, ch in enumerate(media_cols):
        with cols[i % 3]:
            manual_pct[ch] = st.slider(
                clean(ch),
                0,
                100,
                int(100 / len(media_cols)),
                1
            )

    total_pct = sum(manual_pct.values())

    if total_pct == 0:
        st.error("A soma dos percentuais precisa ser maior que zero.")
        st.stop()

    channel_budget = {
        ch: budget_total * manual_pct[ch] / total_pct
        for ch in media_cols
    }

    st.markdown("**Distribuição semanal por canal (%)**")
    st.caption("Ajuste como o budget de cada canal é distribuído ao longo das semanas.")

    weekly_spend_plan = {}

    for ch in media_cols:
        st.markdown(f"**{clean(ch)}** — budget total: R$ {channel_budget[ch]:,.0f}")

        wcols = st.columns(n_weeks)
        wpct = []

        for w in range(n_weeks):
            with wcols[w]:
                wpct.append(
                    st.number_input(
                        week_labels[w],
                        min_value=0,
                        max_value=100,
                        value=int(100 / n_weeks),
                        step=1,
                        key=f"{ch}_{w}"
                    )
                )

        total_wpct = sum(wpct) if sum(wpct) > 0 else 1

        weekly_spend_plan[ch] = np.array(
            [channel_budget[ch] * p / total_wpct for p in wpct]
        )

else:
    with st.spinner("Optimizing 2D allocation channel x week... this may take a few seconds"):
        weekly_spend_plan = optimize_2d(
            budget_total,
            coef_original,
            n_weeks,
            step_size
        )

# === ALLOCATION SUMMARY ===
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
    alloc_df["Share (%)"] = (
        alloc_df["Q1 Total (R$)"] / alloc_df["Q1 Total (R$)"].sum() * 100
    )
else:
    alloc_df["Share (%)"] = 0

alloc_df = alloc_df.sort_values("Q1 Total (R$)", ascending=False).reset_index(drop=True)

col_a, col_b = st.columns([1, 2])

with col_a:
    st.dataframe(
        alloc_df.style.format({
            "Q1 Total (R$)": "R$ {:,.0f}",
            "Share (%)": "{:.1f}%"
        }),
        use_container_width=True
    )

with col_b:
    fig_alloc = px.bar(
        alloc_df,
        x="Canal",
        y="Q1 Total (R$)",
        color="Canal",
        title="Budget Q1 por canal",
        text_auto=".2s",
        category_orders={"Canal": alloc_df["Canal"].tolist()}
    )

    st.plotly_chart(fig_alloc, use_container_width=True)

# === WEEKLY HEATMAP ===
spend_matrix = pd.DataFrame(
    {clean(ch): weekly_spend_plan[ch] for ch in media_cols},
    index=week_labels
)

ordered_channels = alloc_df["Canal"].tolist()
spend_matrix = spend_matrix[ordered_channels]

fig_heat = px.imshow(
    spend_matrix.T,
    text_auto=".2s",
    aspect="auto",
    title="Investimento semanal por canal (R$)",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_heat, use_container_width=True)

# === SIMULATION ===
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

# === KPIS ===
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
r1.metric("ROI 2025 Budget Novo", f"{projected_roi_new_budget:.2f}x")
r2.metric("ROI Mídia Total / Budget 2025", f"{projected_roi_total_media:.2f}x")

# === ROI BY CHANNEL ===
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
        "Efeito Carryover 2024 (R$)": carry_resp,
        "Incremental do Budget 2025 (R$)": new_resp,
        "Efeito Total Mídia (R$)": total_resp,
        "ROI 2025": new_resp / total_spend if total_spend > 0 else 0,
        "ROI Total / Budget 2025": total_resp / total_spend if total_spend > 0 else 0
    })

roi_sim_df = pd.DataFrame(roi_sim_rows).sort_values(
    "ROI 2025",
    ascending=False
)

col_r1, col_r2 = st.columns([1.2, 1.8])

with col_r1:
    st.dataframe(
        roi_sim_df.style.format({
            "Investimento 2025 (R$)": "R$ {:,.0f}",
            "Efeito Carryover 2024 (R$)": "R$ {:,.0f}",
            "Incremental do Budget 2025 (R$)": "R$ {:,.0f}",
            "Efeito Total Mídia (R$)": "R$ {:,.0f}",
            "ROI 2025": "{:.2f}x",
            "ROI Total / Budget 2025": "{:.2f}x"
        }),
        use_container_width=True
    )

with col_r2:
    fig_roi = px.bar(
        roi_sim_df,
        x="Canal",
        y="ROI 2025",
        color="Canal",
        title="ROI 2025 por canal: incremental do budget novo / investimento 2025",
        text_auto=".2f",
        category_orders={"Canal": roi_sim_df["Canal"].tolist()}
    )

    fig_roi.add_hline(
        y=1,
        line_dash="dash",
        line_color="red",
        annotation_text="Break-even"
    )

    st.plotly_chart(fig_roi, use_container_width=True)

# === WEEKLY SALES DECOMPOSITION ===
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
    value_vars=[
        "baseline",
        "carryover_media_effect",
        "new_media_effect",
        "projected_revenue"
    ],
    var_name="Metrica",
    value_name="Valor"
)

decomp_long["Metrica"] = decomp_long["Metrica"].map({
    "baseline": "Baseline Prophet + MMM Controls",
    "carryover_media_effect": "Carryover de Investimentos 2024",
    "new_media_effect": "Incremental do Budget 2025",
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
fig_sales.update_layout(legend_title_text="Component")

st.plotly_chart(fig_sales, use_container_width=True)

# === STACKED DECOMPOSITION ===
st.subheader("3.1 Decomposição empilhada: baseline + carryover + budget 2025")

stack_long = decomp_df.melt(
    id_vars="date",
    value_vars=[
        "baseline",
        "carryover_media_effect",
        "new_media_effect"
    ],
    var_name="Componente",
    value_name="Valor"
)

stack_long["Componente"] = stack_long["Componente"].map({
    "baseline": "Baseline",
    "carryover_media_effect": "Carryover 2024",
    "new_media_effect": "Incremental Budget 2025"
})

fig_stack = px.area(
    stack_long,
    x="date",
    y="Valor",
    color="Componente",
    title="Revenue Decomposition: Baseline vs Carryover vs New Budget"
)

fig_stack.update_yaxes(tickformat=",.0f")
fig_stack.update_xaxes(title="Date")
fig_stack.update_layout(legend_title_text="Componente")

st.plotly_chart(fig_stack, use_container_width=True)

# === WEEKLY MEDIA CONTRIBUTION BY CHANNEL ===
st.subheader("3.2 Contribuição semanal de mídia por canal")

total_effect_cols = [
    f"{ch.replace('_spend_brl', '')}_total_media_effect"
    for ch in media_cols
]

carryover_effect_cols = [
    f"{ch.replace('_spend_brl', '')}_carryover_effect"
    for ch in media_cols
]

new_effect_cols = [
    f"{ch.replace('_spend_brl', '')}_new_investment_effect"
    for ch in media_cols
]

total_media_long = simulation.melt(
    id_vars="date",
    value_vars=total_effect_cols,
    var_name="Canal",
    value_name="Efeito Total Mídia (R$)"
)

total_media_long["Canal"] = (
    total_media_long["Canal"]
    .str.replace("_total_media_effect", "")
    .str.replace("_", " ")
    .str.title()
)

total_order = (
    total_media_long
    .groupby("Canal")["Efeito Total Mídia (R$)"]
    .sum()
    .sort_values(ascending=False)
    .index
    .tolist()
)

fig_total_media_weekly = px.area(
    total_media_long,
    x="date",
    y="Efeito Total Mídia (R$)",
    color="Canal",
    title="Weekly Total Media Effect by Channel: Carryover + New Budget",
    category_orders={"Canal": total_order}
)

fig_total_media_weekly.update_yaxes(tickformat=",.0f")
fig_total_media_weekly.update_xaxes(title="Date")
fig_total_media_weekly.update_layout(legend_title_text="Canal")

st.plotly_chart(fig_total_media_weekly, use_container_width=True)

tab_carry, tab_new = st.tabs(["Carryover 2024 por canal", "Incremental Budget 2025 por canal"])

with tab_carry:
    carryover_long = simulation.melt(
        id_vars="date",
        value_vars=carryover_effect_cols,
        var_name="Canal",
        value_name="Carryover 2024 (R$)"
    )

    carryover_long["Canal"] = (
        carryover_long["Canal"]
        .str.replace("_carryover_effect", "")
        .str.replace("_", " ")
        .str.title()
    )

    carry_order = (
        carryover_long
        .groupby("Canal")["Carryover 2024 (R$)"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig_carry = px.area(
        carryover_long,
        x="date",
        y="Carryover 2024 (R$)",
        color="Canal",
        title="Weekly Carryover Effect from Previous Investments",
        category_orders={"Canal": carry_order}
    )

    fig_carry.update_yaxes(tickformat=",.0f")
    fig_carry.update_xaxes(title="Date")
    st.plotly_chart(fig_carry, use_container_width=True)

with tab_new:
    new_long = simulation.melt(
        id_vars="date",
        value_vars=new_effect_cols,
        var_name="Canal",
        value_name="Incremental Budget 2025 (R$)"
    )

    new_long["Canal"] = (
        new_long["Canal"]
        .str.replace("_new_investment_effect", "")
        .str.replace("_", " ")
        .str.title()
    )

    new_order = (
        new_long
        .groupby("Canal")["Incremental Budget 2025 (R$)"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig_new = px.area(
        new_long,
        x="date",
        y="Incremental Budget 2025 (R$)",
        color="Canal",
        title="Weekly Incremental Effect from 2025 Budget",
        category_orders={"Canal": new_order}
    )

    fig_new.update_yaxes(tickformat=",.0f")
    fig_new.update_xaxes(title="Date")
    st.plotly_chart(fig_new, use_container_width=True)

# === MEDIA CONTRIBUTION SHARE ===
st.subheader("3.3 Participação da contribuição de mídia (%)")

share_metric = st.radio(
    "Métrica de contribuição",
    [
        "Efeito Total Mídia",
        "Carryover 2024",
        "Incremental Budget 2025"
    ],
    horizontal=True
)

if share_metric == "Efeito Total Mídia":
    share_df_source = total_media_long.rename(columns={"Efeito Total Mídia (R$)": "Valor"})
elif share_metric == "Carryover 2024":
    share_df_source = carryover_long.rename(columns={"Carryover 2024 (R$)": "Valor"})
else:
    share_df_source = new_long.rename(columns={"Incremental Budget 2025 (R$)": "Valor"})

media_contribution_pct = (
    share_df_source
    .groupby("Canal")["Valor"]
    .sum()
)

if media_contribution_pct.sum() != 0:
    media_contribution_pct = media_contribution_pct / media_contribution_pct.sum() * 100
else:
    media_contribution_pct = media_contribution_pct * 0

media_contribution_df = (
    media_contribution_pct
    .sort_values(ascending=True)
    .reset_index()
)

media_contribution_df.columns = ["Canal", "Contribution (%)"]

fig_media_pct = px.bar(
    media_contribution_df,
    x="Contribution (%)",
    y="Canal",
    orientation="h",
    title=f"Media Contribution Share (%) - {share_metric}",
    text="Contribution (%)"
)

fig_media_pct.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_media_pct.update_layout(xaxis_title="Contribution (%)", yaxis_title="")
fig_media_pct.update_xaxes(
    range=[0, max(media_contribution_df["Contribution (%)"].max() * 1.15, 1)]
)

st.plotly_chart(fig_media_pct, use_container_width=True)

# === ADSTOCK ===
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
    .str.replace("_adstock_total", "")
    .str.replace("_", " ")
    .str.title()
)

fig_ads = px.line(
    ads_long,
    x="date",
    y="Adstock Total",
    color="Canal",
    markers=True,
    title="Adstock Total: Carryover + Spend 2025"
)

fig_ads.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig_ads, use_container_width=True)

with st.expander("Comparar adstock total vs carryover only"):
    adstock_compare_rows = []

    for ch in media_cols:
        cn = ch.replace("_spend_brl", "")

        tmp_total = pd.DataFrame({
            "date": simulation["date"],
            "Canal": clean(ch),
            "Tipo": "Adstock Total",
            "Adstock": simulation[f"{cn}_adstock_total"]
        })

        tmp_carry = pd.DataFrame({
            "date": simulation["date"],
            "Canal": clean(ch),
            "Tipo": "Carryover Only",
            "Adstock": simulation[f"{cn}_adstock_carryover_only"]
        })

        adstock_compare_rows.append(tmp_total)
        adstock_compare_rows.append(tmp_carry)

    adstock_compare_df = pd.concat(adstock_compare_rows)

    selected_adstock_channel = st.selectbox(
        "Canal para comparar",
        [clean(ch) for ch in media_cols]
    )

    adstock_compare_filtered = adstock_compare_df[
        adstock_compare_df["Canal"] == selected_adstock_channel
    ]

    fig_ads_compare = px.line(
        adstock_compare_filtered,
        x="date",
        y="Adstock",
        color="Tipo",
        markers=True,
        title=f"Adstock Total vs Carryover Only - {selected_adstock_channel}"
    )

    fig_ads_compare.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig_ads_compare, use_container_width=True)

# === SATURATION ===
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
    .str.replace("_saturation_total", "")
    .str.replace("_", " ")
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

fig_sat.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Alta saturação 80%")
fig_sat.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Saturação média 50%")

st.plotly_chart(fig_sat, use_container_width=True)

# === DEBUG BASELINE ===
with st.expander("Debug Prophet Baseline"):
    st.dataframe(
        future_controls.assign(baseline=baseline_weekly),
        use_container_width=True
    )

# === DEBUG HILL PARAMS ===
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

# === WEEKLY PLAN ===
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

# === HISTORICAL ROI ===
st.subheader("6. ROI histórico estimado pelo modelo")

st.dataframe(
    roi_table.style.format({
        "Incremental Revenue": "R$ {:,.0f}",
        "Spend": "R$ {:,.0f}",
        "Contribution (%)": "{:.1f}%",
        "ROI": "{:.2f}"
    }),
    use_container_width=True
)

# === RESPONSE CURVES ===
st.subheader("7. Response curves por canal")

st.caption(
    "A curva mostra o efeito incremental do novo spend sobre o carryover existente. "
    "O carryover anterior continua existindo mesmo com spend novo igual a zero."
)

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
        "Carryover 2024 (R$)": resp_carryover,
        "Incremental Budget 2025 (R$)": resp_new,
        "Saturacao (%)": sat_grid,
        "Canal": clean(ch)
    }))

if curve_data:
    curve_df = pd.concat(curve_data)

    tab1, tab2, tab3 = st.tabs([
        "Incremental Budget 2025",
        "Efeito Total Mídia",
        "Saturação"
    ])

    with tab1:
        fig_curve_new = px.line(
            curve_df,
            x="Spend Novo (R$)",
            y="Incremental Budget 2025 (R$)",
            color="Canal",
            title="Response Curve - Incremental do Budget 2025 vs Spend Novo"
        )

        pts = []

        for ch in selected_channels:
            beta = get_beta(ch, coef_original)
            spend_w1 = weekly_spend_plan[ch][0]
            spend_arr = np.zeros(n_weeks)
            spend_arr[0] = spend_w1

            _, _, new_effect, ads_total, _ = split_media_effect(
                spend_arr=spend_arr,
                channel=ch,
                beta=beta,
                n_weeks=n_weeks
            )

            pts.append({
                "Spend Novo (R$)": spend_w1,
                "Incremental Budget 2025 (R$)": new_effect[0],
                "Canal": clean(ch)
            })

        pts_df = pd.DataFrame(pts)

        fig_pts = px.scatter(
            pts_df,
            x="Spend Novo (R$)",
            y="Incremental Budget 2025 (R$)",
            color="Canal"
        )

        for trace in fig_pts.data:
            trace.marker.size = 12
            trace.marker.symbol = "diamond"
            fig_curve_new.add_trace(trace)

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

# === DOWNLOAD ===
st.subheader("8. Exportar simulação")

csv = weekly_plan.to_csv(index=False).encode("utf-8")

st.download_button(
    "Baixar plano semanal em CSV",
    csv,
    "mmm_q1_2025_weekly_plan.csv",
    "text/csv"
)
