# Marketing Mix Modeling | Varejo360

## Business Problem

A Varejo360 é um dos maiores e-commerces de moda e acessórios do Brasil, com atuação nacional e milhões de clientes ativos.

A empresa investe constantemente em múltiplos canais de mídia — como TV, Google, Meta, OOH e Influenciadores — e precisava responder uma pergunta estratégica para o próximo ciclo orçamentário:

> "Qual é a contribuição incremental de cada canal de mídia sobre as vendas e qual o ROI de cada investimento?"

O objetivo do projeto foi desenvolver um framework de Marketing Mix Modeling (MMM) para:
- medir incrementalidade;
- calcular ROI por canal;
- modelar carryover effects;
- analisar saturation effects;
- otimizar alocação de budget.

---

# Dataset

O dataset possui granularidade semanal, com 3 anos de histórico, e contém informações de:

### Media Channels
- TV
- Google Search
- Google Display
- Meta Ads
- OOH
- Influenciadores
- CRM

### Controls & External Variables
- Economic Confidence Index
- Competitor Promotion Index
- Holiday Weeks
- Black Friday

### Target Variable
- Revenue (BRL)

---

# Methodology

O projeto foi desenvolvido utilizando uma abordagem clássica de Marketing Mix Modeling (MMM).

## Main Steps

- Exploratory Data Analysis (EDA)
- Time Series Decomposition
- Adstock Transformation
- Hill Saturation Curves
- Ridge Regression
- ROI Decomposition
- Budget Optimization

---

# Adstock

Adstock foi utilizado para modelar carryover effects e persistência temporal da mídia.

$$
Adstock_t = x_t + \theta x_{t-1} + \theta^2 x_{t-2}
$$

---

# Hill Saturation

Hill Curves foram utilizadas para modelar diminishing returns e saturation effects.

$$
f(x)=\frac{x^\alpha}{\gamma^\alpha+x^\alpha}
$$

---

# Final Product

O projeto resultou em uma aplicação interativa para:
- simulação de budget;
- análise de ROI;
- projeção de receita;
- otimização de mídia;
- visualização de response curves;
- análise de carryover effects.

## Streamlit App

🚀 https://mmm-varejo360.streamlit.app/

---

## Next Steps

As próximas evoluções do projeto incluem o desenvolvimento de uma otimização não míope, capaz de considerar não apenas o retorno incremental dentro do Q1, mas também o carryover gerado para as semanas seguintes.

Atualmente, o otimizador aloca budget considerando o impacto dentro das 13 semanas simuladas. Como alguns canais possuem efeito prolongado de adstock, especialmente TV e OOH, parte do retorno gerado por investimentos feitos no final do Q1 pode aparecer apenas no início do Q2.

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Prophet
- Streamlit
- Plotly
- Matplotlib
- Seaborn
