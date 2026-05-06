
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="MMM Budget Simulator",
    layout="wide"
)

with open("mmm_artifacts.pkl", "rb") as f:
    artifact = pickle.load(f)

df_mmm = artifact["df_mmm"]
media_cols = artifact["media_cols"]
media_features = artifact["media_features"]
adstock_params = artifact["adstock_params"]
hill_params = artifact["hill_params"]
all_results = artifact["all_results"]
X_columns = artifact["X_columns"]

def hill_function(x, k, s):
    x = np.maximum(np.asarray(x), 0)
    return (x ** s) / ((x ** s) + (k ** s) + 1e-9)

def get_hill_s(channel):
    value = hill_params[channel]
    if isinstance(value, dict):
        return value.get("s", value.get("alpha", 2.0))
    return value

def get_beta(channel, coef_original):
    hill_col = f"{channel}_hill"
    idx = X_columns.index(hill_col)
    return coef_original[idx]

def response_function(spend, channel, beta):
    alpha = adstock_params[channel]
    s = get_hill_s(channel)
    adstock_col = f"{channel}_adstock"
    if adstock_col in df_mmm.columns:
        k = df_mmm[adstock_col].median()
    else:
        k = df_mmm[channel].median()
    adstocked = spend / (1 - alpha)
    hill_value = hill_function(adstocked, k=k, s=s)
    return beta * hill_value

def optimize_weekly_budget(weekly_budget, coef_original, step_size=5000):
    allocation = {channel: 0 for channel in media_cols}
    remaining = weekly_budget
    while remaining > 0:
        step = min(step_size, remaining)
        best_channel = None
        best_gain = -np.inf
        for channel in media_cols:
            beta = get_beta(channel, coef_original)
            current_spend = allocation[channel]
            current_response = response_function(current_spend, channel, beta)
            next_response = response_function(current_spend + step, channel, beta)
            marginal_gain = next_response - current_response
            if marginal_gain > best_gain:
                best_gain = marginal_gain
                best_channel = channel
        allocation[best_channel] += step
        remaining -= step
    return allocation

def clean_channel_name(channel):
    return (
        channel
        .replace("_spend_brl", "")
        .replace("_hill", "")
        .replace("_", " ")
        .title()
    )

st.sidebar.title("Inputs do Simulador")

model_name = st.sidebar.selectbox(
    "Modelo",
    list(all_results.keys()),
    index=list(all_results.keys()).index("Ridge") if "Ridge" in all_results.keys() else 0
)

res = all_results[model_name]
coef_original = res["coef_original"]
roi_table = res["roi_table"]

budget_total = st.sidebar.number_input("Budget total Q1 2025", min_value=0, value=3000000, step=100000)
baseline_weekly = st.sidebar.number_input("Baseline semanal", min_value=0, value=int(df_mmm["revenue_brl"].tail(13).mean()) if "revenue_brl" in df_mmm.columns else 1000000, step=50000)
allocation_mode = st.sidebar.radio("Modo de alocação", ["Otimizado por Marginal ROI", "Manual"])
step_size = st.sidebar.number_input("Granularidade da otimização", min_value=1000, value=5000, step=1000)

weeks = pd.date_range("2025-01-06", "2025-03-31", freq="W-MON")
n_weeks = len(weeks)
weekly_budget = budget_total / n_weeks if n_weeks > 0 else 0

st.title("MMM Interactive Budget Simulator")
st.caption("Simulador de budget, ROI, response curve e projeção de vendas — Q1 2025")

st.subheader("1. Alocação de budget por canal")

if allocation_mode == "Manual":
    cols = st.columns(3)
    manual_pct = {}
    for i, channel in enumerate(media_cols):
        with cols[i % 3]:
            manual_pct[channel] = st.slider(clean_channel_name(channel), min_value=0, max_value=100, value=int(100 / len(media_cols)), step=1)
    total_pct = sum(manual_pct.values())
    if total_pct == 0:
        st.error("A soma dos percentuais precisa ser maior que zero.")
        st.stop()
    weekly_allocation = {channel: weekly_budget * manual_pct[channel] / total_pct for channel in media_cols}
else:
    weekly_allocation = optimize_weekly_budget(weekly_budget=weekly_budget, coef_original=coef_original, step_size=step_size)

allocation_df = pd.DataFrame({
    "channel": [clean_channel_name(c) for c in weekly_allocation.keys()],
    "weekly_spend": list(weekly_allocation.values())
})
allocation_df["q1_spend"] = allocation_df["weekly_spend"] * n_weeks
allocation_df["share"] = allocation_df["q1_spend"] / allocation_df["q1_spend"].sum()

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(allocation_df.style.format({"weekly_spend": "R$ {:,.0f}", "q1_spend": "R$ {:,.0f}", "share": "{:.1%}"}), use_container_width=True)
with col_b:
    fig_alloc = px.bar(allocation_df, x="channel", y="q1_spend", title="Budget Q1 por canal")
    st.plotly_chart(fig_alloc, use_container_width=True)

simulation = pd.DataFrame({"date": weeks})
simulation["baseline"] = baseline_weekly
simulation["weekly_budget"] = weekly_budget
incremental_total = np.zeros(n_weeks)

for channel in media_cols:
    beta = get_beta(channel, coef_original)
    weekly_spend = weekly_allocation[channel]
    response = response_function(weekly_spend, channel, beta)
    clean_name = channel.replace("_spend_brl", "")
    simulation[f"{clean_name}_spend"] = weekly_spend
    simulation[f"{clean_name}_incremental"] = response
    incremental_total += response

simulation["incremental_revenue"] = incremental_total
simulation["projected_revenue"] = simulation["baseline"] + simulation["incremental_revenue"]

st.subheader("2. Resultado projetado")
total_revenue = simulation["projected_revenue"].sum()
total_incremental = simulation["incremental_revenue"].sum()
projected_roi = total_incremental / budget_total if budget_total > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Budget Total", f"R$ {budget_total:,.0f}")
kpi2.metric("Venda Projetada", f"R$ {total_revenue:,.0f}")
kpi3.metric("Receita Incremental", f"R$ {total_incremental:,.0f}")
kpi4.metric("ROI Projetado", f"{projected_roi:.2f}")

st.subheader("3. Projeção semanal de vendas")
sales_long = simulation.melt(id_vars="date", value_vars=["baseline", "incremental_revenue", "projected_revenue"], var_name="metric", value_name="value")
fig_sales = px.line(sales_long, x="date", y="value", color="metric", markers=True, title="Baseline, Incremental e Venda Projetada")
st.plotly_chart(fig_sales, use_container_width=True)

st.subheader("4. Plano semanal sugerido")
display_cols = ["date", "baseline", "weekly_budget", "incremental_revenue", "projected_revenue"]
spend_cols = [c for c in simulation.columns if c.endswith("_spend")]
weekly_plan = simulation[display_cols + spend_cols].copy()
st.dataframe(weekly_plan.style.format({col: "R$ {:,.0f}" for col in weekly_plan.columns if col != "date"}), use_container_width=True)

st.subheader("5. ROI histórico estimado pelo modelo")
st.dataframe(roi_table.style.format({"Incremental Revenue": "R$ {:,.0f}", "Spend": "R$ {:,.0f}", "Contribution (%)": "{:.1f}%", "ROI": "{:.2f}"}), use_container_width=True)

st.subheader("6. Response curves")
selected_channels = st.multiselect("Canais para visualizar", media_cols, default=media_cols)
curve_data = []
for channel in selected_channels:
    beta = get_beta(channel, coef_original)
    max_spend = max(df_mmm[channel].max() * 1.5, weekly_allocation[channel] * 2, 1)
    spend_grid = np.linspace(0, max_spend, 200)
    response_grid = response_function(spend_grid, channel, beta)
    curve_data.append(pd.DataFrame({"spend": spend_grid, "incremental_revenue": response_grid, "channel": clean_channel_name(channel)}))

if curve_data:
    curve_df = pd.concat(curve_data)
    fig_curve = px.line(curve_df, x="spend", y="incremental_revenue", color="channel", title="Response Curve por canal")
    current_points = []
    for channel in selected_channels:
        beta = get_beta(channel, coef_original)
        spend = weekly_allocation[channel]
        response = response_function(spend, channel, beta)
        current_points.append({"spend": spend, "incremental_revenue": response, "channel": clean_channel_name(channel)})
    current_points_df = pd.DataFrame(current_points)
    fig_points = px.scatter(current_points_df, x="spend", y="incremental_revenue", color="channel")
    for trace in fig_points.data:
        fig_curve.add_trace(trace)
    st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("7. Exportar simulação")
csv = weekly_plan.to_csv(index=False).encode("utf-8")
st.download_button(label="Baixar plano semanal em CSV", data=csv, file_name="mmm_q1_2025_weekly_plan.csv", mime="text/csv")
