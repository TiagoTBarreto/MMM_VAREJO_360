import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='MMM Budget Simulator', layout='wide')

with open('mmm_artifacts.pkl', 'rb') as f:
    artifact = pickle.load(f)

df_mmm         = artifact['df_mmm']
media_cols     = artifact['media_cols']
adstock_params = artifact['adstock_params']
hill_params    = artifact['hill_params']
all_results    = artifact['all_results']
X_columns      = artifact['X_columns']

initial_adstock = {}
for ch in media_cols:
    alpha = adstock_params[ch]
    spend_series = df_mmm[ch].values
    adstock_hist = np.zeros(len(spend_series))
    adstock_hist[0] = spend_series[0]
    for t in range(1, len(spend_series)):
        adstock_hist[t] = spend_series[t] + alpha * adstock_hist[t - 1]
    initial_adstock[ch] = adstock_hist[-1]

def hill_function(x, k, s):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    return (x ** s) / ((x ** s) + (k ** s) + 1e-9)

def get_hill_params(channel):
    value = hill_params[channel]
    if isinstance(value, dict):
        return value.get('k', df_mmm[channel].median()), value.get('s', 2.0)
    return df_mmm[channel].median(), float(value)

def get_beta(channel, coef_original):
    hill_col = f'{channel}_hill'
    idx = X_columns.index(hill_col)
    return coef_original[idx]

def simulate_adstock_series(weekly_spend, channel, n_weeks):
    alpha = adstock_params[channel]
    adstock_series = np.zeros(n_weeks)
    adstock_series[0] = weekly_spend + alpha * initial_adstock[channel]
    for t in range(1, n_weeks):
        adstock_series[t] = weekly_spend + alpha * adstock_series[t - 1]
    return adstock_series

def response_series(weekly_spend, channel, beta, n_weeks):
    k, s = get_hill_params(channel)
    adstock_series = simulate_adstock_series(weekly_spend, channel, n_weeks)
    hill_values = hill_function(adstock_series, k=k, s=s)
    return beta * hill_values

def optimize_weekly_budget(weekly_budget, coef_original, n_weeks, step_size=5000):
    allocation = {ch: 0.0 for ch in media_cols}
    remaining  = weekly_budget
    while remaining > 1e-6:
        step = min(step_size, remaining)
        best_channel, best_gain = None, -np.inf
        for ch in media_cols:
            beta = get_beta(ch, coef_original)
            cur  = response_series(allocation[ch],        ch, beta, n_weeks).sum()
            nxt  = response_series(allocation[ch] + step, ch, beta, n_weeks).sum()
            gain = nxt - cur
            if gain > best_gain:
                best_gain, best_channel = gain, ch
        allocation[best_channel] += step
        remaining -= step
    return allocation

def clean(channel):
    return channel.replace('_spend_brl','').replace('_hill','').replace('_',' ').title()

st.sidebar.title('Inputs do Simulador')

model_name = st.sidebar.selectbox(
    'Modelo', list(all_results.keys()),
    index=list(all_results.keys()).index('Ridge') if 'Ridge' in all_results else 0
)

res           = all_results[model_name]
coef_original = res['coef_original']
roi_table     = res['roi_table']
contributions = res['contributions'].copy()
contributions['date'] = pd.to_datetime(contributions['date'])

budget_total    = st.sidebar.number_input('Budget total Q1 2025', min_value=0, value=3_000_000, step=100_000)
allocation_mode = st.sidebar.radio('Modo de alocacao', ['Otimizado por Marginal ROI', 'Manual'])
step_size       = st.sidebar.number_input('Granularidade da otimizacao', min_value=1000, value=5000, step=1000)

weeks         = pd.date_range('2025-01-06', '2025-03-31', freq='W-MON')
n_weeks       = len(weeks)
weekly_budget = budget_total / n_weeks if n_weeks > 0 else 0

last_baseline   = contributions['baseline'].tail(13).values
baseline_weekly = np.array([last_baseline[i % len(last_baseline)] for i in range(n_weeks)])

st.title('MMM Interactive Budget Simulator')
st.caption('Simulador de budget, ROI, response curve e projecao de vendas - Q1 2025')

st.subheader('1. Alocacao de budget por canal')

if allocation_mode == 'Manual':
    cols = st.columns(3)
    manual_pct = {}
    for i, ch in enumerate(media_cols):
        with cols[i % 3]:
            manual_pct[ch] = st.slider(clean(ch), 0, 100, int(100/len(media_cols)), 1)
    total_pct = sum(manual_pct.values())
    if total_pct == 0:
        st.error('A soma dos percentuais precisa ser maior que zero.')
        st.stop()
    weekly_allocation = {ch: weekly_budget * manual_pct[ch] / total_pct for ch in media_cols}
else:
    with st.spinner('Otimizando alocacao por marginal ROI...'):
        weekly_allocation = optimize_weekly_budget(weekly_budget, coef_original, n_weeks, step_size)

allocation_df = pd.DataFrame({
    'Canal':         [clean(c) for c in weekly_allocation],
    'Semanal (R$)':  list(weekly_allocation.values()),
    'Q1 Total (R$)': [v * n_weeks for v in weekly_allocation.values()],
})
allocation_df['Share (%)'] = allocation_df['Q1 Total (R$)'] / allocation_df['Q1 Total (R$)'].sum() * 100

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(
        allocation_df.style.format({'Semanal (R$)': 'R$ {:,.0f}', 'Q1 Total (R$)': 'R$ {:,.0f}', 'Share (%)': '{:.1f}%'}),
        use_container_width=True
    )
with col_b:
    fig_alloc = px.bar(allocation_df, x='Canal', y='Q1 Total (R$)', title='Budget Q1 por canal', color='Canal', text_auto='.2s')
    st.plotly_chart(fig_alloc, use_container_width=True)

simulation = pd.DataFrame({'date': weeks})
simulation['baseline'] = baseline_weekly
incremental_total = np.zeros(n_weeks)

for ch in media_cols:
    beta         = get_beta(ch, coef_original)
    weekly_spend = weekly_allocation[ch]
    resp         = response_series(weekly_spend, ch, beta, n_weeks)
    clean_name   = ch.replace('_spend_brl', '')
    simulation[f'{clean_name}_adstock']     = simulate_adstock_series(weekly_spend, ch, n_weeks)
    simulation[f'{clean_name}_spend']       = weekly_spend
    simulation[f'{clean_name}_incremental'] = resp
    incremental_total += resp

simulation['incremental_revenue'] = incremental_total
simulation['projected_revenue']   = simulation['baseline'] + simulation['incremental_revenue']

st.subheader('2. Resultado projetado - Q1 2025')

total_revenue     = simulation['projected_revenue'].sum()
total_baseline    = simulation['baseline'].sum()
total_incremental = simulation['incremental_revenue'].sum()
projected_roi     = total_incremental / budget_total if budget_total > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric('Budget Total',          f'R$ {budget_total:,.0f}')
k2.metric('Baseline Projetado',    f'R$ {total_baseline:,.0f}')
k3.metric('Receita Incremental',   f'R$ {total_incremental:,.0f}')
k4.metric('Venda Total Projetada', f'R$ {total_revenue:,.0f}')
st.metric('ROI Projetado', f'{projected_roi:.2f}x')

st.subheader('3. Projecao semanal de vendas')

sales_long = simulation.melt(
    id_vars='date',
    value_vars=['baseline', 'incremental_revenue', 'projected_revenue'],
    var_name='Metrica', value_name='Valor'
)
label_map = {'baseline': 'Baseline', 'incremental_revenue': 'Incremental de Midia', 'projected_revenue': 'Venda Projetada'}
sales_long['Metrica'] = sales_long['Metrica'].map(label_map)
fig_sales = px.line(sales_long, x='date', y='Valor', color='Metrica', markers=True, title='Baseline variavel + Incremental de midia (adstock real)')
st.plotly_chart(fig_sales, use_container_width=True)

st.subheader('3.1 Adstock acumulado por canal ao longo do Q1')

adstock_cols = [c for c in simulation.columns if c.endswith('_adstock')]
adstock_long = simulation.melt(id_vars='date', value_vars=adstock_cols, var_name='Canal', value_name='Adstock')
adstock_long['Canal'] = adstock_long['Canal'].str.replace('_adstock','').str.replace('_',' ').str.title()
fig_adstock = px.line(adstock_long, x='date', y='Adstock', color='Canal', markers=True, title='Adstock acumulado por canal - Q1 2025')
st.plotly_chart(fig_adstock, use_container_width=True)

st.subheader('4. Plano semanal sugerido')

spend_cols   = [c for c in simulation.columns if c.endswith('_spend')]
display_cols = ['date', 'baseline', 'incremental_revenue', 'projected_revenue'] + spend_cols
weekly_plan  = simulation[display_cols].copy()
st.dataframe(
    weekly_plan.style.format({col: 'R$ {:,.0f}' for col in weekly_plan.columns if col != 'date'}),
    use_container_width=True
)

st.subheader('5. ROI historico estimado pelo modelo')
st.dataframe(
    roi_table.style.format({'Incremental Revenue': 'R$ {:,.0f}', 'Spend': 'R$ {:,.0f}', 'Contribution (%)': '{:.1f}%', 'ROI': '{:.2f}'}),
    use_container_width=True
)

st.subheader('6. Response curves por canal')

selected_channels = st.multiselect('Canais para visualizar', media_cols, default=media_cols)
curve_data = []

for ch in selected_channels:
    beta      = get_beta(ch, coef_original)
    max_spend = max(df_mmm[ch].max() * 1.5, weekly_allocation[ch] * 2, 1)
    grid      = np.linspace(0, max_spend, 300)
    k, s      = get_hill_params(ch)
    alpha     = adstock_params[ch]
    adstock_grid = grid + alpha * initial_adstock[ch]
    resp_grid    = get_beta(ch, coef_original) * hill_function(adstock_grid, k=k, s=s)
    curve_data.append(pd.DataFrame({'Spend': grid, 'Incremental Revenue': resp_grid, 'Canal': clean(ch)}))

if curve_data:
    curve_df  = pd.concat(curve_data)
    fig_curve = px.line(curve_df, x='Spend', y='Incremental Revenue', color='Canal', title='Response Curve - semana 1 do Q1 (com carryover historico)')
    pts = []
    for ch in selected_channels:
        beta  = get_beta(ch, coef_original)
        spend = weekly_allocation[ch]
        k, s  = get_hill_params(ch)
        alpha = adstock_params[ch]
        ads   = spend + alpha * initial_adstock[ch]
        resp  = beta * hill_function(ads, k=k, s=s)
        pts.append({'Spend': spend, 'Incremental Revenue': resp, 'Canal': clean(ch)})
    pts_df  = pd.DataFrame(pts)
    fig_pts = px.scatter(pts_df, x='Spend', y='Incremental Revenue', color='Canal')
    for trace in fig_pts.data:
        fig_curve.add_trace(trace)
    st.plotly_chart(fig_curve, use_container_width=True)

st.subheader('7. Exportar simulacao')
csv = weekly_plan.to_csv(index=False).encode('utf-8')
st.download_button('Baixar plano semanal em CSV', csv, 'mmm_q1_2025_weekly_plan.csv', 'text/csv')
