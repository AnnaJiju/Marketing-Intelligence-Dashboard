# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from utils import load_data, prepare_data, compute_channel_metrics, DATA_DIR

# Page config
st.set_page_config(page_title="Marketing Intelligence Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Global Dark Theme CSS
st.markdown("""
    <style>
    body {background-color: #0b0b0b; color: #e6eef8;}
    .reportview-container, .main, .block-container {background-color: #0b0b0b; color: #e6eef8;}
    .css-1d391kg {background-color: #0b0b0b;}
    .stButton>button {background-color:#1f6feb;color:white;}
    .kpi {
        background-color:#0f1720;
        padding:20px;
        margin:5px;
        border-radius:10px;
        text-align:center;
        min-height:120px;
    }
    .kpi h3 {
        margin-bottom:8px;
        font-size:18px;
        color:#9ae6b4;
    }
    .kpi p {
        font-size:22px;
        font-weight:bold;
        color:#ffffff;
    }
    /* Force sidebar always visible + remove collapse/expand button */
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 300px;
    }
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"

# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner=True)
def get_data():
    cleaned_path = os.path.join(DATA_DIR, "cleaned_data.csv")
    daily_path = os.path.join(DATA_DIR, "cleaned_daily.csv")
    if os.path.exists(cleaned_path) and os.path.exists(daily_path):
        df = pd.read_csv(cleaned_path, parse_dates=['date'])
        daily = pd.read_csv(daily_path, parse_dates=['date'])
    else:
        fb, google, tiktok, biz = load_data()
        df, daily = prepare_data(fb, google, tiktok, biz)
    return df, daily

df, daily = get_data()

st.markdown(
    """
    <h1 style='text-align: center; color: #9ae6b4; font-size:40px;'>
        📊 Marketing Intelligence Dashboard
    </h1>
    <h4 style='text-align: center; color: #a0aec0;'>
        One place for Spend • Revenue • ROAS • Orders • Customers
    </h4>
    <hr style="border:1px solid #333;">
    """,
    unsafe_allow_html=True
)
# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)

platform_options = ["All"] + sorted(df['platform'].dropna().unique().tolist())
platforms = st.sidebar.multiselect("Platform", options=platform_options, default=["All"])

state_list = sorted(df['state'].dropna().unique().tolist())
states = st.sidebar.multiselect("State / Tactic", options=state_list, default=[])

campaign_list = sorted(df['campaign'].dropna().unique().tolist())
campaigns = st.sidebar.multiselect("Campaign", options=campaign_list, default=[])

show_ctr = st.sidebar.checkbox("Show CTR charts", value=True)
show_cpc = st.sidebar.checkbox("Show CPC charts", value=True)

# Apply filters
mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
if "All" not in platforms:
    mask &= df['platform'].isin([p for p in platforms if p != "All"])
if states:
    mask &= df['state'].isin(states)
if campaigns:
    mask &= df['campaign'].isin(campaigns)

filtered = df.loc[mask].copy()
daily_filtered = daily[(daily['date'] >= pd.to_datetime(date_range[0])) &
                       (daily['date'] <= pd.to_datetime(date_range[1]))].copy()

if filtered.empty:
    st.warning("No data for selected filters. Adjust date range / platform / state / campaign.")
    st.stop()

# -------------------------
# KPIs
# -------------------------
total_spend = filtered['spend'].sum()
total_attributed_revenue = filtered['attributed_revenue'].sum()
business_total_revenue = filtered['total_revenue'].sum()
orders_sum = filtered['orders'].sum()
new_customers_cols = [c for c in filtered.columns if 'new' in c and 'customer' in c]
new_customers_sum = int(filtered[new_customers_cols].sum(axis=1).sum()) if new_customers_cols else 0

roas_overall = (total_attributed_revenue / total_spend) if total_spend > 0 else np.nan
cpa = (total_spend / new_customers_sum) if new_customers_sum > 0 else np.nan
cpo = (total_spend / orders_sum) if orders_sum > 0 else np.nan

# KPI Layout (first row: 5 tiles)
row1 = st.columns(5)
with row1[0]:
    st.markdown(f"<div class='kpi'><h3>Total Spend</h3><p>${total_spend:,.0f}</p></div>", unsafe_allow_html=True)
with row1[1]:
    st.markdown(f"<div class='kpi'><h3>Attributed Revenue</h3><p>${total_attributed_revenue:,.0f}</p></div>", unsafe_allow_html=True)
with row1[2]:
    roas_display = f"{roas_overall:.2f}" if not np.isnan(roas_overall) else "N/A"
    st.markdown(f"<div class='kpi'><h3>ROAS</h3><p>{roas_display}</p></div>", unsafe_allow_html=True)
with row1[3]:
    st.markdown(f"<div class='kpi'><h3>Total Orders</h3><p>{int(orders_sum):,}</p></div>", unsafe_allow_html=True)
with row1[4]:
    st.markdown(f"<div class='kpi'><h3>New Customers</h3><p>{int(new_customers_sum):,}</p></div>", unsafe_allow_html=True)

# KPI Layout (second row: 3 tiles)
row2 = st.columns(3)
with row2[0]:
    st.markdown(f"<div class='kpi'><h3>CPA</h3><p>{f'${cpa:,.2f}' if not np.isnan(cpa) else 'N/A'}</p></div>", unsafe_allow_html=True)
with row2[1]:
    st.markdown(f"<div class='kpi'><h3>CPO</h3><p>{f'${cpo:,.2f}' if not np.isnan(cpo) else 'N/A'}</p></div>", unsafe_allow_html=True)
with row2[2]:
    st.markdown(f"<div class='kpi'><h3>Total Revenue</h3><p>${business_total_revenue:,.0f}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Chart Grid Layout
# -------------------------

# Row 1: Revenue by State, Top Campaigns, Spend Distribution
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.subheader("Revenue by State")
    state_rev = filtered.groupby('state', as_index=False)['total_revenue'].sum().sort_values('total_revenue', ascending=False).head(8)
    fig_state = px.bar(state_rev, x='state', y='total_revenue', template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_state, use_container_width=True)
with c2:
    st.subheader("Top Campaigns")
    camp_tbl = filtered.groupby(['platform','campaign'], as_index=False)['attributed_revenue'].sum().sort_values('attributed_revenue', ascending=False).head(6)
    fig_top_c = px.bar(camp_tbl.sort_values('attributed_revenue'),
                       x='attributed_revenue', y='campaign',
                       orientation='h', color='platform', template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_top_c, use_container_width=True)
with c3:
    st.subheader("Spend Distribution")
    plat_spend = filtered.groupby('platform', as_index=False)['spend'].sum()
    fig_donut = px.pie(plat_spend, names='platform', values='spend', hole=0.5, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_donut, use_container_width=True)

# Row 2: Revenue vs Spend, Daily ROAS, ROAS by Platform
c1, c2, c3 = st.columns([1.5, 1, 1])
with c1:
    st.subheader("Revenue vs Spend (Daily)")
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['total_revenue'], mode='lines', name='Revenue', fill='tozeroy'))
    fig_rs.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['spend'], mode='lines', name='Spend', fill='tozeroy'))
    fig_rs.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_rs, use_container_width=True)
with c2:
    st.subheader("Daily ROAS")
    if daily_filtered['spend'].sum() > 0:
        daily_filtered['roas'] = np.where(daily_filtered['spend']>0, daily_filtered['attributed_revenue']/daily_filtered['spend'], np.nan)
        fig_roas_t = px.line(daily_filtered, x='date', y='roas', template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_roas_t, use_container_width=True)
with c3:
    st.subheader("ROAS by Platform")
    plat_metrics = compute_channel_metrics(filtered)
    fig_roas_p = px.bar(plat_metrics, x='platform', y='roas', text=plat_metrics['roas'].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"), template=PLOTLY_TEMPLATE)
    fig_roas_p.update_traces(textposition='outside')
    st.plotly_chart(fig_roas_p, use_container_width=True)

# Row 3: Campaign Scatter, CPC/CPM, Orders Trend
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.subheader("Campaign Efficiency")
    camp_df = filtered.groupby(['platform','campaign'], as_index=False).agg(
        spend=('spend','sum'),
        attributed_revenue=('attributed_revenue','sum'),
        impressions=('impression','sum'),
        clicks=('click','sum')
    )
    if not camp_df.empty:
        fig_camp = px.scatter(camp_df, x='spend', y='attributed_revenue', size='impressions', color='platform', hover_data=['campaign'], template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_camp, use_container_width=True)
with c2:
    st.subheader("Cost Metrics (CPC / CPM)")
    daily_costs = daily_filtered.copy()
    daily_costs['cpc'] = np.where(daily_costs['click']>0, daily_costs['spend']/daily_costs['click'], np.nan)
    daily_costs['cpm'] = np.where(daily_costs['impression']>0, daily_costs['spend']/(daily_costs['impression']/1000), np.nan)
    fig_cost = go.Figure()
    if show_cpc:
        fig_cost.add_trace(go.Scatter(x=daily_costs['date'], y=daily_costs['cpc'], name='CPC'))
    if show_ctr:
        fig_cost.add_trace(go.Scatter(x=daily_costs['date'], y=daily_costs['cpm'], name='CPM'))
    fig_cost.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_cost, use_container_width=True)
with c3:
    st.subheader("Orders Trend")
    fig_orders = px.bar(daily_filtered, x='date', y='orders', template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_orders, use_container_width=True)

# Row 4: Stacked Attributed vs Organic Revenue, State Matrix, Campaign Table
c1, c2, c3 = st.columns([1.5, 1, 0.8])
with c1:
    st.subheader("Attributed vs Organic Revenue")
    daily_filtered['organic_revenue'] = (daily_filtered['total_revenue'] - daily_filtered['attributed_revenue']).clip(lower=0)
    stacked_df = daily_filtered[['date','attributed_revenue','organic_revenue']].melt(id_vars='date', var_name='type', value_name='amount')
    fig_stack = px.area(stacked_df, x='date', y='amount', color='type', template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_stack, use_container_width=True)
with c2:
    st.subheader("State Performance Matrix")
    state_table = filtered.groupby('state', as_index=False).agg(
        spend=('spend','sum'),
        attributed_revenue=('attributed_revenue','sum'),
        total_revenue=('total_revenue','sum')
    )
    state_table['roas'] = np.where(state_table['spend']>0, state_table['attributed_revenue']/state_table['spend'], np.nan)
    fig_state_matrix = px.scatter(state_table, x='spend', y='total_revenue', size='roas', color='state', template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_state_matrix, use_container_width=True)
with c3:
    st.subheader("Top Campaigns Table")
    st.dataframe(camp_df.sort_values('attributed_revenue', ascending=False).head(12).reset_index(drop=True))


# ---- Summary Report ----
st.markdown("---")
st.subheader("📌 Summary Report")

summary_points = []

# Spend vs Revenue
if total_spend > 0 and total_attributed_revenue > 0:
    roas_val = total_attributed_revenue / total_spend
    if roas_val >= 3:
        summary_points.append("✅ Marketing spend is generating excellent returns (ROAS above 3).")
    elif roas_val >= 1.5:
        summary_points.append("⚖️ Marketing efficiency is decent (ROAS between 1.5 and 3). Further optimization possible.")
    else:
        summary_points.append("⚠️ ROAS is low (below 1.5). Spend may not be efficient, consider reallocating budget.")

# Orders and customers
if orders_sum > 0:
    summary_points.append(f"📦 A total of **{orders_sum:,.0f} orders** were recorded during the selected period.")
if new_customers_sum > 0:
    summary_points.append(f"🎯 Marketing campaigns brought in **{new_customers_sum:,.0f} new customers**.")

# Platform insights
plat_metrics = compute_channel_metrics(filtered)
if not plat_metrics.empty:
    top_platform = plat_metrics.sort_values("roas", ascending=False).iloc[0]
    summary_points.append(
        f"🏆 **{top_platform['platform']}** shows the best efficiency with ROAS of {top_platform['roas']:.2f}."
    )

# Business revenue
if business_total_revenue > 0:
    summary_points.append(
        f"💵 Total business revenue reached **${business_total_revenue:,.0f}** during the selected period."
    )

# Show as bullet points
if summary_points:
    for point in summary_points:
        st.markdown(f"- {point}")
else:
    st.info("No significant insights available for the selected filters.")


st.markdown("---")
# -------------------------
# Exports & Notes
# -------------------------
st.subheader("Export & Notes")
colx1, colx2 = st.columns([1, 2])
with colx1:
    st.download_button("Download Filtered Data", filtered.to_csv(index=False).encode('utf-8'), "filtered_data.csv", "text/csv")
    st.download_button("Download Campaign Summary", camp_df.to_csv(index=False).encode('utf-8'), "campaigns.csv", "text/csv")
with colx2:
    st.markdown("""
    **Notes & assumptions**
    - ROAS = attributed_revenue / spend (where spend > 0).  
    - Organic revenue = total_revenue - attributed_revenue.  
    - Data cleaned: missing values filled, negatives clipped, duplicates removed.  
    - Use sidebar filters to drill down by platform / state / campaign.
    """)

st.markdown("<div style='color:#7f8c8d;font-size:12px'>Built with Streamlit • Compact Dark Dashboard</div>", unsafe_allow_html=True)
