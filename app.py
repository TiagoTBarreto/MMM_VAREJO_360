import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from itertools import product

st.set_page_config(page_title='MMM Budget Simulator', layout='wide')

with open('mmm_artifacts.pkl', 'rb') as f:
    artifact = pickle.load(f)

df_mmm         = artifact['df_mmm']
media_cols     = artifact['media_cols']
adstock_params = artifact['adstock_params']
hill_params    = artifact['hill_params']
all_results    = artifact['all_results']
X_columns      = artifact['X_columns']

# === CARRYOVER INICIAL ===
initial_adstock = {}
for ch in media_cols:
    alpha = adstock_params[ch]
    sv = df_mmm[ch].values
    ah = np.zeros(len(sv))
    ah[0] = sv[0]
    for t in range(1, len(sv)):
        ah[t] = sv[t] + alpha * ah[t-1]
    initial_adstock[ch] = ah[-1]

# === FUNCTIONS ===
def hill_function(x, k, s):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    return (x**s) / ((x**s) + (k**s) + 1e-9)

def get_hill_params(channel):
    value = hill_params[channel]
    if isinstance(value, dict):
        return value.get('k', df_mmm[channel].median()), value.get('s', 2.0)
    return df_mmm[channel].median(), float(value)

def get_beta(channel, coef_original):
    idx = X_columns.index(f'{channel}_hill')
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
        ads[t] = spend_arr[t] + alpha * ads[t-1]
    return ads

def response_from_adstock(adstock_series, channel, beta):
    k, s = get_hill_params(channel)
    return beta * hill_function(adstock_series, k=k, s=s)

def saturation_pct(adstock_series, channel):
    k, s = get_hill_params(channel)
    return hill_function(adstock_series, k=k, s=s) * 100

def optimize_2d(budget_total, coef_original, n_weeks, step_size=5000):
    # Otimizacao 2D: aloca step para (canal, semana) com maior ganho marginal
    spend = {ch: np.zeros(n_weeks) for ch in media_cols}
    adstock = {ch: simulate_adstock_series(spend[ch], ch, n_weeks) for ch in media_cols}
    remaining = budget_total
    while remaining > 1e-6:
        step = min(step_size, remaining)
        best, best_gain = (None, None), -np.inf
        for ch in media_cols:
            beta = get_beta(ch, coef_original)
            for w in range(n_weeks):
                cur_resp  = response_from_adstock(adstock[ch], ch, beta).sum()
                new_spend = spend[ch].copy()
                new_spend[w] += step
                new_ads   = simulate_adstock_series(new_spend, ch, n_weeks)
                new_resp  = response_from_adstock(new_ads, ch, beta).sum()
                gain = new_resp - cur_resp
                if gain > best_gain:
                    best_gain = gain
                    best = (ch, w)
        ch_best, w_best = best
        spend[ch_best][w_best] += step
        adstock[ch_best] = simulate_adstock_series(spend[ch_best], ch_best, n_weeks)
        remaining -= step
    return spend

def clean(ch):
    return ch.replace('_spend_brl','').replace('_hill','').replace('_',' ').title()

# === SIDEBAR ===
st.sidebar.title('Inputs do Simulador')
model_name = st.sidebar.selectbox('Modelo', list(all_results.keys()),
    index=list(all_results.keys()).index('Ridge') if 'Ridge' in all_results else 0)
res           = all_results[model_name]
coef_original = res['coef_original']
roi_table     = res['roi_table']
contributions = res['contributions'].copy()
contributions['date'] = pd.to_datetime(contributions['date'])

budget_total    = st.sidebar.number_input('Budget total Q1 2025', min_value=0, value=3_000_000, step=100_000)
allocation_mode = st.sidebar.radio('Modo de alocacao', ['Otimizado (2D)', 'Manual'])
step_size       = st.sidebar.number_input('Granularidade otimizacao', min_value=1000, value=10000, step=1000)

# === DATE RANGE ===
weeks         = pd.date_range('2025-01-06', '2025-03-31', freq='W-MON')
n_weeks       = len(weeks)
weekly_budget = budget_total / n_weeks if n_weeks > 0 else 0
last_baseline   = contributions['baseline'].tail(13).values
baseline_weekly = np.array([last_baseline[i % len(last_baseline)] for i in range(n_weeks)])
week_labels = [str(w.date()) for w in weeks]

st.title('MMM Interactive Budget Simulator')
st.caption('Simulador de budget, ROI, response curve e projecao de vendas - Q1 2025')

# === ALLOCATION ===
st.subheader('1. Alocacao de budget por canal')

if allocation_mode == 'Manual':
    st.markdown('**Distribuicao por canal (%)**')
    cols = st.columns(3)
    manual_pct = {}
    for i, ch in enumerate(media_cols):
        with cols[i % 3]:
            manual_pct[ch] = st.slider(clean(ch), 0, 100, int(100/len(media_cols)), 1)
    total_pct = sum(manual_pct.values())
    if total_pct == 0:
        st.error('A soma dos percentuais precisa ser maior que zero.')
        st.stop()
    channel_budget = {ch: budget_total * manual_pct[ch] / total_pct for ch in media_cols}
    st.markdown('**Distribuicao semanal por canal (%)**')
    st.caption('Ajuste como o budget de cada canal e distribuido ao longo das semanas')
    weekly_spend_plan = {}
    for ch in media_cols:
        st.markdown(f'*{clean(ch)}* — budget total: R$ {channel_budget[ch]:,.0f}')
        wcols = st.columns(n_weeks)
        wpct = []
        for w in range(n_weeks):
            with wcols[w]:
                wpct.append(st.number_input(week_labels[w], min_value=0, max_value=100,
                    value=int(100/n_weeks), step=1, key=f'{ch}_{w}'))
        total_wpct = sum(wpct) if sum(wpct) > 0 else 1
        weekly_spend_plan[ch] = np.array([channel_budget[ch] * p / total_wpct for p in wpct])
else:
    with st.spinner('Otimizando alocacao 2D (canal x semana)... isso pode levar alguns segundos'):
        weekly_spend_plan = optimize_2d(budget_total, coef_original, n_weeks, step_size)

# Resumo de alocacao
alloc_rows = []
for ch in media_cols:
    total_ch = weekly_spend_plan[ch].sum()
    alloc_rows.append({'Canal': clean(ch), 'Q1 Total (R$)': total_ch,
        'Share (%)': 0})
alloc_df = pd.DataFrame(alloc_rows)
alloc_df['Share (%)'] = alloc_df['Q1 Total (R$)'] / alloc_df['Q1 Total (R$)'].sum() * 100

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(alloc_df.style.format({'Q1 Total (R$)': 'R$ {:,.0f}', 'Share (%)': '{:.1f}%'}), use_container_width=True)
with col_b:
    fig_alloc = px.bar(alloc_df, x='Canal', y='Q1 Total (R$)', color='Canal',
        title='Budget Q1 por canal', text_auto='.2s')
    st.plotly_chart(fig_alloc, use_container_width=True)

# Heatmap de spend semanal
spend_matrix = pd.DataFrame({clean(ch): weekly_spend_plan[ch] for ch in media_cols}, index=week_labels)
fig_heat = px.imshow(spend_matrix.T, text_auto='.2s', aspect='auto',
    title='Investimento semanal por canal (R$)', color_continuous_scale='Blues')
st.plotly_chart(fig_heat, use_container_width=True)

# === SIMULATION ===
simulation = pd.DataFrame({'date': weeks})
simulation['baseline'] = baseline_weekly
incremental_total = np.zeros(n_weeks)

for ch in media_cols:
    beta        = get_beta(ch, coef_original)
    spend_arr   = weekly_spend_plan[ch]
    ads         = simulate_adstock_series(spend_arr, ch, n_weeks)
    resp        = response_from_adstock(ads, ch, beta)
    sat         = saturation_pct(ads, ch)
    cn          = ch.replace('_spend_brl','')
    simulation[f'{cn}_spend']       = spend_arr
    simulation[f'{cn}_adstock']     = ads
    simulation[f'{cn}_incremental'] = resp
    simulation[f'{cn}_saturation']  = sat
    incremental_total += resp

simulation['incremental_revenue'] = incremental_total
simulation['projected_revenue']   = simulation['baseline'] + simulation['incremental_revenue']

# === KPIS ===
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

# === ROI POR CANAL ===
st.subheader('2.1 ROI simulado por canal')
roi_sim_rows = []
for ch in media_cols:
    beta      = get_beta(ch, coef_original)
    spend_arr = weekly_spend_plan[ch]
    ads       = simulate_adstock_series(spend_arr, ch, n_weeks)
    resp      = response_from_adstock(ads, ch, beta)
    total_spend = spend_arr.sum()
    total_resp  = resp.sum()
    roi_sim_rows.append({
        'Canal': clean(ch),
        'Investimento Q1 (R$)': total_spend,
        'Receita Incremental (R$)': total_resp,
        'ROI Simulado': total_resp / total_spend if total_spend > 0 else 0
    })
roi_sim_df = pd.DataFrame(roi_sim_rows).sort_values('ROI Simulado', ascending=False)
col_r1, col_r2 = st.columns([1, 2])
with col_r1:
    st.dataframe(roi_sim_df.style.format({
        'Investimento Q1 (R$)': 'R$ {:,.0f}',
        'Receita Incremental (R$)': 'R$ {:,.0f}',
        'ROI Simulado': '{:.2f}x'
    }), use_container_width=True)
with col_r2:
    fig_roi = px.bar(roi_sim_df, x='Canal', y='ROI Simulado', color='Canal',
        title='ROI simulado por canal com investimento sugerido', text_auto='.2f')
    fig_roi.add_hline(y=1, line_dash='dash', line_color='red', annotation_text='Break-even')
    st.plotly_chart(fig_roi, use_container_width=True)

# === WEEKLY SALES ===
st.subheader('3. Projecao semanal de vendas')
sales_long = simulation.melt(id_vars='date',
    value_vars=['baseline','incremental_revenue','projected_revenue'],
    var_name='Metrica', value_name='Valor')
sales_long['Metrica'] = sales_long['Metrica'].map({
    'baseline':'Baseline','incremental_revenue':'Incremental de Midia','projected_revenue':'Venda Projetada'})
fig_sales = px.line(sales_long, x='date', y='Valor', color='Metrica', markers=True,
    title='Baseline variavel + Incremental de midia (adstock real)')
st.plotly_chart(fig_sales, use_container_width=True)

# === ADSTOCK ===
st.subheader('3.1 Adstock efetivo por canal ao longo do Q1')
adstock_cols = [c for c in simulation.columns if c.endswith('_adstock')]
ads_long = simulation.melt(id_vars='date', value_vars=adstock_cols, var_name='Canal', value_name='Adstock')
ads_long['Canal'] = ads_long['Canal'].str.replace('_adstock','').str.replace('_',' ').str.title()
fig_ads = px.line(ads_long, x='date', y='Adstock', color='Canal', markers=True,
    title='Adstock efetivo (spend + carryover) por canal - Q1 2025')
st.plotly_chart(fig_ads, use_container_width=True)

# === SATURACAO SEMANAL ===
st.subheader('3.2 Saturacao semanal por canal (%)')
st.caption('% de saturacao da hill function aplicada sobre o adstock efetivo (spend + carryover). 100% = saturacao total.')
sat_cols = [c for c in simulation.columns if c.endswith('_saturation')]
sat_long = simulation.melt(id_vars='date', value_vars=sat_cols, var_name='Canal', value_name='Saturacao (%)')
sat_long['Canal'] = sat_long['Canal'].str.replace('_saturation','').str.replace('_',' ').str.title()
fig_sat = px.line(sat_long, x='date', y='Saturacao (%)', color='Canal', markers=True,
    title='Saturacao semanal por canal (%)', range_y=[0,100])
fig_sat.add_hline(y=80, line_dash='dash', line_color='orange', annotation_text='Alta saturacao (80%)')
fig_sat.add_hline(y=50, line_dash='dot', line_color='green', annotation_text='Saturacao media (50%)')
st.plotly_chart(fig_sat, use_container_width=True)

# === WEEKLY PLAN ===
st.subheader('4. Plano semanal sugerido')
spend_cols   = [c for c in simulation.columns if c.endswith('_spend')]
display_cols = ['date','baseline','incremental_revenue','projected_revenue'] + spend_cols
weekly_plan  = simulation[display_cols].copy()
st.dataframe(weekly_plan.style.format({col: 'R$ {:,.0f}' for col in weekly_plan.columns if col != 'date'}), use_container_width=True)

# === ROI HISTORICO ===
st.subheader('5. ROI historico estimado pelo modelo')
st.dataframe(roi_table.style.format({
    'Incremental Revenue': 'R$ {:,.0f}', 'Spend': 'R$ {:,.0f}',
    'Contribution (%)': '{:.1f}%', 'ROI': '{:.2f}'}), use_container_width=True)

# === RESPONSE CURVES ===
st.subheader('6. Response curves por canal')
st.caption('Eixo X = Adstock efetivo (spend novo + carryover historico da semana 1). O marcador mostra onde o investimento sugerido posiciona cada canal na curva.')
selected_channels = st.multiselect('Canais para visualizar', media_cols, default=media_cols)
curve_data = []
for ch in selected_channels:
    beta      = get_beta(ch, coef_original)
    k, s      = get_hill_params(ch)
    alpha     = adstock_params[ch]
    carry     = initial_adstock[ch]
    max_raw   = max(df_mmm[ch].max() * 1.5, weekly_spend_plan[ch].max() * 2, 1)
    grid_spend = np.linspace(0, max_raw, 300)
    grid_ads   = grid_spend + alpha * carry
    resp_grid  = beta * hill_function(grid_ads, k=k, s=s)
    sat_grid   = hill_function(grid_ads, k=k, s=s) * 100
    curve_data.append(pd.DataFrame({
        'Adstock Efetivo (R$)': grid_ads,
        'Spend Novo (R$)': grid_spend,
        'Receita Incremental (R$)': resp_grid,
        'Saturacao (%)': sat_grid,
        'Canal': clean(ch)
    }))

if curve_data:
    curve_df = pd.concat(curve_data)
    tab1, tab2 = st.tabs(['Receita Incremental', 'Saturacao'])
    with tab1:
        fig_curve = px.line(curve_df, x='Adstock Efetivo (R$)', y='Receita Incremental (R$)',
            color='Canal', title='Response Curve - Receita incremental vs Adstock efetivo (semana 1 Q1)')
        pts = []
        for ch in selected_channels:
            beta  = get_beta(ch, coef_original)
            k, s  = get_hill_params(ch)
            alpha = adstock_params[ch]
            spend_w1 = weekly_spend_plan[ch][0]
            ads_w1   = spend_w1 + alpha * initial_adstock[ch]
            resp_w1  = beta * hill_function(ads_w1, k=k, s=s)
            pts.append({'Adstock Efetivo (R$)': ads_w1, 'Receita Incremental (R$)': resp_w1,
                'Canal': clean(ch), 'Spend novo': f'R$ {spend_w1:,.0f}', 'Carryover': f'R$ {alpha*initial_adstock[ch]:,.0f}'})
        pts_df = pd.DataFrame(pts)
        fig_pts = px.scatter(pts_df, x='Adstock Efetivo (R$)', y='Receita Incremental (R$)',
            color='Canal', hover_data=['Spend novo','Carryover'], size_max=12)
        for trace in fig_pts.data:
            trace.marker.size = 12
            trace.marker.symbol = 'diamond'
            fig_curve.add_trace(trace)
        st.plotly_chart(fig_curve, use_container_width=True)
    with tab2:
        fig_sat2 = px.line(curve_df, x='Adstock Efetivo (R$)', y='Saturacao (%)',
            color='Canal', title='Saturacao vs Adstock efetivo (semana 1 Q1)', range_y=[0,100])
        fig_sat2.add_hline(y=80, line_dash='dash', line_color='orange', annotation_text='Alta saturacao')
        fig_sat2.add_hline(y=50, line_dash='dot', line_color='green', annotation_text='Saturacao media')
        pts2 = []
        for ch in selected_channels:
            k, s  = get_hill_params(ch)
            alpha = adstock_params[ch]
            spend_w1 = weekly_spend_plan[ch][0]
            ads_w1   = spend_w1 + alpha * initial_adstock[ch]
            sat_w1   = hill_function(ads_w1, k=k, s=s) * 100
            pts2.append({'Adstock Efetivo (R$)': ads_w1, 'Saturacao (%)': sat_w1, 'Canal': clean(ch)})
        pts2_df = pd.DataFrame(pts2)
        fig_pts2 = px.scatter(pts2_df, x='Adstock Efetivo (R$)', y='Saturacao (%)', color='Canal')
        for trace in fig_pts2.data:
            trace.marker.size = 12
            trace.marker.symbol = 'diamond'
            fig_sat2.add_trace(trace)
        st.plotly_chart(fig_sat2, use_container_width=True)

# === DOWNLOAD ===
st.subheader('7. Exportar simulacao')
csv = weekly_plan.to_csv(index=False).encode('utf-8')
st.download_button('Baixar plano semanal em CSV', csv, 'mmm_q1_2025_weekly_plan.csv', 'text/csv')