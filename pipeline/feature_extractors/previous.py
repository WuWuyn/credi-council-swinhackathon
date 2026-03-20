import pandas as pd
import numpy as np
import os

def extract_previous_features(sk_id_curr, data_dir):
    """
    Extracts features derived from previous_application, installments_payments, 
    and POS_CASH_balance for a single SK_ID_CURR.
    """
    prev_path = os.path.join(data_dir, 'previous_application.csv')
    inst_path = os.path.join(data_dir, 'installments_payments.csv')
    pos_path  = os.path.join(data_dir, 'POS_CASH_balance.csv')
    
    if not os.path.exists(prev_path):
        return {}

    # Load and filter Data
    prev = pd.read_csv(prev_path)
    prev = prev[prev['SK_ID_CURR'] == sk_id_curr].copy()
    if prev.empty:
        return {}
        
    sk_id_prevs = prev['SK_ID_PREV'].unique()
    
    # Optional Installments data
    inst_prev_last = {}
    if os.path.exists(inst_path):
        inst = pd.read_csv(inst_path)
        inst = inst[inst['SK_ID_PREV'].isin(sk_id_prevs)]
        if not inst.empty:
            inst_prev_last_s = inst.groupby('SK_ID_PREV')['AMT_PAYMENT'].sum()
            inst_prev_last = inst_prev_last_s.to_dict()
            
    # Optional POS data
    pos_prev_last_dict = {}
    pos_prev_cnt_instalment = {}
    if os.path.exists(pos_path):
        pos = pd.read_csv(pos_path)
        pos = pos[pos['SK_ID_PREV'].isin(sk_id_prevs)]
        if not pos.empty:
            idx = pos.groupby(['SK_ID_PREV'])['MONTHS_BALANCE'].idxmax()
            pos_last = pos[['SK_ID_PREV','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE']].loc[idx.values]
            pos_last['INSTAL_LEFT_RATIO'] = pos_last['CNT_INSTALMENT_FUTURE'] / (pos_last['CNT_INSTALMENT'].replace(0, np.nan))
            pos_prev_last_dict = dict(zip(pos_last['SK_ID_PREV'], pos_last['INSTAL_LEFT_RATIO']))
            pos_prev_cnt_instalment = dict(zip(pos_last['SK_ID_PREV'], pos_last['CNT_INSTALMENT']))

    # --- Previous Applications Filtering ---
    prev = prev.loc[prev['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y']
    for f_ in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        # We need to simulate the notebook logic carefully:
        if f_ in prev.columns:
            prev[f_] = prev[f_].where(prev[f_] <= 360000, np.nan)
            
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT'].replace(0, np.nan)
    prev['AMT_DIFF_CREAPP'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['AMT_DIFF_CREDIT_GOODS'] = prev['AMT_CREDIT'] - prev['AMT_GOODS_PRICE']
    prev['AMT_CREDIT_GOODS_PERC'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE'].replace(0, np.nan)
    prev['AMT_PAY_YEAR'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY'].replace(0, np.nan)
    prev['DAYS_TOTAL'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DUE']
    prev['DAYS_TOTAL2'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DUE']
    prev['DAYS_END_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
    
    prev['CNT_PAYMENT_DIFF'] = prev['CNT_PAYMENT'] - prev['SK_ID_PREV'].map(pos_prev_cnt_instalment)
    
    # Ignoring DEFAULTED calculation for now as it relies on more subsets

    # Recent application
    idx_recent = prev['DAYS_DECISION'].idxmax()
    if pd.isna(idx_recent):
        prev_recent = prev.iloc[-1:] # Fallback
    else:
        prev_recent = prev.loc[[idx_recent]]
    prev_recent_dict = {}
    for c in prev_recent.columns:
        if c not in ['SK_ID_PREV', 'SK_ID_CURR']:
            v = prev_recent[c].values[0]
            if type(v) in [str, bool]: 
                 pass # usually factorized, skipping categorical in dict for simplicity
            else:
                 prev_recent_dict[f"prev_recent_{c}"] = v
                 
    # Dummies replacement
    prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == 'object']
    for f_ in prev_cat_features:
        if prev[f_].nunique(dropna=False) <= 2:
            prev[f_], _ = pd.factorize(prev[f_])
        else:
            prev = pd.concat([prev, pd.get_dummies(prev[f_], prefix=f_)], axis=1)
            del prev[f_]
            
    avg_feats = [f_ for f_ in prev.columns.values if ('DAYS' in f_) or ('RATE' in f_) or ('AMT' in f_)]
    for f_ in avg_feats:
        prev[f_] = prev[f_].where(prev[f_] <= 300000, np.nan)
        
    res = {}
    # avg
    for f_ in avg_feats:
        if f_ in prev.columns and not pd.isna(prev[f_].mean()):
            res[f"prev_avg_{f_}"] = prev[f_].mean().item()
            
    # max
    max_feats = [f_ for f_ in prev.columns.values if ('DAYS' in f_) or ('AMT' in f_)]
    for f_ in max_feats:
         if f_ in prev.columns and not pd.isna(prev[f_].max()):
            res[f"prev_max_{f_}"] = prev[f_].max().item()

    # min
    if 'DAYS_DECISION' in prev.columns and not pd.isna(prev['DAYS_DECISION'].min()):
         res['prev_min_DAYS_DECISION'] = prev['DAYS_DECISION'].min().item()
         
    # sum
    nosum_feats = ['SK_ID_CURR','SK_ID_PREV','DAYS_TOTAL','DAYS_TOTAL2','DAYS_FIRST_DRAWING',
                   'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
                   'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
                   'AMT_CREDIT_GOODS_PERC','APP_CREDIT_PERC']
    sum_feats = [f_ for f_ in prev.columns.values if f_ not in nosum_feats]
    for f_ in sum_feats:
        if f_ in prev.columns and not pd.isna(prev[f_].sum()):
            res[f"prev_sum_{f_}"] = prev[f_].sum().item()
            
    # prev_active
    prev_active = prev.loc[(prev['DAYS_LAST_DUE'].isnull()) & (prev['DAYS_LAST_DUE_1ST_VERSION'] > 0)].copy()
    if not prev_active.empty:
        prev_active['AMT_LEFT'] = prev_active['AMT_ANNUITY'] * prev_active['DAYS_LAST_DUE_1ST_VERSION'] / 365.25
        prev_active['AMT_PAID'] = prev_active['SK_ID_PREV'].map(inst_prev_last).fillna(0)
        prev_active['AMT_OWE'] = (prev_active['AMT_CREDIT'] - prev_active['AMT_DOWN_PAYMENT'].fillna(0)) * (1 + prev_active['RATE_INTEREST_PRIVILEGED'].fillna(0))
        prev_active['AMT_LEFT2'] = (prev_active['AMT_OWE'] - prev_active['AMT_PAID']).clip(lower=0)
        prev_active['LEFT_RATIO'] = prev_active['SK_ID_PREV'].map(pos_prev_last_dict)
        prev_active['AMT_LEFT3'] = prev_active['AMT_CREDIT'] * prev_active['LEFT_RATIO']
        prev_active['AMT_PAY_YEAR_LEFT'] = prev_active['AMT_LEFT'] / prev_active['AMT_ANNUITY'].replace(0, np.nan)
        
        active_sum_feats = [f_ for f_ in prev_active.columns.values if ('AMT' in f_)]
        for f_ in active_sum_feats:
             if f_ in prev_active.columns and not pd.isna(prev_active[f_].sum()):
                 res[f"prev_active_sum_{f_}"] = prev_active[f_].sum().item()
                 
    # Merge recent dict mapping numpy types appropriately
    for k, v in prev_recent_dict.items():
        if pd.isna(v): continue
        if np.isscalar(v) and hasattr(v, 'item'):
            res[k] = v.item()
        elif isinstance(v, (int, float)):
            res[k] = v

    return res

if __name__ == "__main__":
    import json
    data_dir = 'c:/Users/Admin/Desktop/MAS Swinburne/credi-council-swinhackathon/home-credit-default-risk'
    print(json.dumps(extract_previous_features(100001, data_dir), indent=2))
