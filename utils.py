# utils.py
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_data():
    """Load raw CSVs from data/ folder (no heavy cleaning)."""
    fb = pd.read_csv(os.path.join(DATA_DIR, "Facebook.csv"))
    google = pd.read_csv(os.path.join(DATA_DIR, "Google.csv"))
    tiktok = pd.read_csv(os.path.join(DATA_DIR, "TikTok.csv"))
    biz = pd.read_csv(os.path.join(DATA_DIR, "business.csv"))
    # Normalize column names to lowercase trimmed strings
    for df in [fb, google, tiktok, biz]:
        df.columns = [str(c).strip().lower() for c in df.columns]
    return fb, google, tiktok, biz

def _prep_channel(df, platform_name):
    """Standardize each channel dataframe to common schema and types, add platform."""
    df = df.copy()
    rename_map = {}
    for c in df.columns:
        c_low = c.lower()
        if 'date' in c_low: rename_map[c] = 'date'
        if 'tactic' in c_low: rename_map[c] = 'tactic'
        if c_low == 'state': rename_map[c] = 'state'
        if 'campaign' in c_low: rename_map[c] = 'campaign'
        if 'impress' in c_low: rename_map[c] = 'impression'
        # clicks or click
        if 'click' in c_low and 'clicks' not in rename_map.values():
            rename_map[c] = 'click'
        if 'spend' in c_low: rename_map[c] = 'spend'
        if 'attribut' in c_low or ('attributed' in c_low) or ('attributed revenue' in c_low):
            rename_map[c] = 'attributed_revenue'
        # fallback revenue column mapping
        if c_low == 'revenue' and 'attributed_revenue' not in rename_map.values():
            rename_map[c] = 'attributed_revenue'
    df = df.rename(columns=rename_map)

    # Coerce types
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['impression', 'click', 'spend', 'attributed_revenue']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # Add platform column
    df['platform'] = platform_name

    # Ensure optional columns exist
    if 'campaign' not in df.columns:
        df['campaign'] = '(unknown)'
    if 'state' not in df.columns:
        df['state'] = '(unknown)'
    if 'tactic' not in df.columns:
        df['tactic'] = '(unknown)'

    return df

def prepare_data(fb, google, tiktok, biz, save_clean=True):
    """
    Full cleaning pipeline:
    - standardize channels -> combine into `marketing`
    - merge with business on date -> `merged` (granular rows)
    - aggregate daily -> `daily` (unique dates)
    - clip negatives, dedupe, fill na, compute derived metrics
    - saves cleaned_data.csv and cleaned_daily.csv in data/
    Returns (merged, daily)
    """
    # per-channel cleaning
    fb2 = _prep_channel(fb, "Facebook")
    google2 = _prep_channel(google, "Google")
    tiktok2 = _prep_channel(tiktok, "TikTok")

    marketing = pd.concat([fb2, google2, tiktok2], ignore_index=True, sort=False)
    marketing = marketing.drop_duplicates().reset_index(drop=True)

    # business
    biz = biz.copy().drop_duplicates().reset_index(drop=True)
    biz['date'] = pd.to_datetime(biz.get('date'), errors='coerce')

    # harmonize business revenue/orders names
    # handle variants like 'total revenue', 'total_revenue', 'totalrevenue'
    biz_cols = list(biz.columns)
    if 'total revenue' in biz_cols:
        biz = biz.rename(columns={'total revenue': 'total_revenue'})
    if 'totalrevenue' in biz_cols:
        biz = biz.rename(columns={'totalrevenue': 'total_revenue'})
    if 'total_revenue' not in biz.columns:
        for c in biz_cols:
            if 'revenue' in c and c != 'attributed_revenue':
                biz = biz.rename(columns={c: 'total_revenue'})
                break

    if 'orders' not in biz.columns:
        for c in biz_cols:
            if 'order' in c:
                biz = biz.rename(columns={c: 'orders'})
                break

    biz['total_revenue'] = pd.to_numeric(biz.get('total_revenue', 0), errors='coerce').fillna(0)
    biz['orders'] = pd.to_numeric(biz.get('orders', 0), errors='coerce').fillna(0)

    # merge marketing granularity (platform x campaign x date) with business daily
    merged = pd.merge(marketing, biz, on='date', how='left', suffixes=('','_biz'))

    # numeric fill/convert
    for col in ['total_revenue', 'orders', 'impression', 'click', 'spend', 'attributed_revenue']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)
        else:
            merged[col] = 0

    # clip negatives (no negative spend/impressions)
    for col in ['spend', 'impression', 'click', 'attributed_revenue', 'total_revenue', 'orders']:
        if col in merged.columns:
            merged[col] = merged[col].clip(lower=0)

    # derived metrics at row granularity
    merged['ctr'] = np.where(merged['impression'] > 0, merged['click'] / merged['impression'], 0)
    merged['cpc'] = np.where(merged['click'] > 0, merged['spend'] / merged['click'], np.nan)
    # attribution ratio: portion of business revenue attributed to marketing (if both present)
    merged['attrib_ratio'] = np.where(merged['total_revenue'] > 0, merged['attributed_revenue'] / merged['total_revenue'], 0)

    # create daily aggregated dataset (unique dates)
    daily = merged.groupby('date', as_index=False).agg({
        'spend': 'sum',
        'impression': 'sum',
        'click': 'sum',
        'attributed_revenue': 'sum',
        'total_revenue': 'sum',
        'orders': 'sum'
    })

    # ensure continuous date index across min->max
    if not daily.empty:
        full_dates = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
        daily = daily.set_index('date').reindex(full_dates, fill_value=0).reset_index().rename(columns={'index':'date'})

    # compute daily derived metrics
    daily['ctr'] = np.where(daily['impression'] > 0, daily['click'] / daily['impression'], 0)
    daily['cpc'] = np.where(daily['click'] > 0, daily['spend'] / daily['click'], np.nan)
    daily['roas'] = np.where(daily['spend'] > 0, daily['attributed_revenue'] / daily['spend'], np.nan)

    # save cleaned artifacts
    if save_clean:
        os.makedirs(DATA_DIR, exist_ok=True)
        merged.to_csv(os.path.join(DATA_DIR, "cleaned_data.csv"), index=False)
        daily.to_csv(os.path.join(DATA_DIR, "cleaned_daily.csv"), index=False)

    return merged, daily

def compute_channel_metrics(df):
    """
    Aggregation by platform (keeps same name for backward compatibility).
    Returns platform-level metrics.
    """
    agg = df.groupby('platform', as_index=False).agg(
        spend=('spend','sum'),
        revenue=('attributed_revenue','sum'),
        impressions=('impression','sum'),
        clicks=('click','sum')
    )
    agg['ctr'] = np.where(agg['impressions']>0, agg['clicks'] / agg['impressions'], 0)
    agg['cpc'] = np.where(agg['clicks']>0, agg['spend'] / agg['clicks'], np.nan)
    agg['roas'] = np.where(agg['spend']>0, agg['revenue'] / agg['spend'], np.nan)
    return agg[['platform','spend','revenue','impressions','clicks','ctr','cpc','roas']]
