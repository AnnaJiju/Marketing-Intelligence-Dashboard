# 📊 Marketing Intelligence Dashboard

An interactive BI dashboard built using **Streamlit** to analyze the performance of marketing campaigns (Facebook, Google, TikTok) in relation to business outcomes like revenue, orders, and profit.  

This dashboard helps stakeholders answer key questions:
- Which platform gives the best ROAS (Return On Ad Spend)?
- How are marketing spend and impressions converting into orders and revenue?
- What are the trends of profit margin and revenue per order over time?
- How do campaign-level metrics compare across platforms?

---

## 🚀 Live Demo

https://annajiju-marketing-intelligence-dashboard-app-odztro.streamlit.app/
> *Note: If you're seeing an error, the dashboard might be temporarily down or there might be issues with deployment. See the Troubleshooting section below.*

---

## 🧰 Features

- Load and merge data from multiple sources: **Facebook**, **Google**, **TikTok**, and **Business performance**.  
- Data cleaning and transformation:
  - Standardize date formats  
  - Normalize metrics (impressions, clicks, spend, revenue)  
  - Handle missing values and duplicates  
- Derived KPIs:
  - CTR (Click-Through Rate)  
  - CPC (Cost per Click)  
  - ROAS (Return on Ad Spend)  
  - CPM (Cost per 1,000 Impressions)  
  - Revenue per Order  
  - Profit Margin  
- Interactive Visualizations:
  - Time-series trends of spend, revenue, profit  
  - Platform comparisons  
  - Campaign-level performance  
  - Business outcomes vs. marketing metrics  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/marketing-intelligence-dashboard.git
cd marketing-intelligence-dashboard
```
### 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the app
streamlit run app.py

