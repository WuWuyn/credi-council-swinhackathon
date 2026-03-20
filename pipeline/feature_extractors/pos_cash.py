import pandas as pd
import numpy as np
import os

def extract_pos_cash_features(sk_id_curr, data_dir):
    """
    Extracts features derived from POS_CASH_balance for a single SK_ID_CURR.
    """
    pos_path  = os.path.join(data_dir, 'POS_CASH_balance.csv')
    if not os.path.exists(pos_path):
        return {}

    pos = pd.read_csv(pos_path)
    pos = pos[pos['SK_ID_CURR'] == sk_id_curr].copy()
    if pos.empty:
        return {}

    # recent data
    idx = pos['MONTHS_BALANCE'].idxmax()
    pos_recent = pos[['MONTHS_BALANCE','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                      'NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']].loc[[idx]]
    # Factorize NAME_CONTRACT_STATUS (since we only operate on 1 row, we'll assign it 0 assuming factorize starts at 0, 
    # but the notebook ran factorize on the whole dataset. 
    # To be extremely mathematically consistent with the notebook's single-row processing, 
    # we just ignore categorical replacement for `recent_` and let LightGBM deal with the absent column if we skip it,
    # or just record the string. LightGBM models might expect an integer here. Let's cast to a dummy int or keep it out).
    pos_recent_dict = {}
    for c in pos_recent.columns:
         v = pos_recent[c].values[0]
         if c == 'NAME_CONTRACT_STATUS': # Notebook factorizes this
             # We won't try to guess the global factorized ID of this status. We'll skip it unless it's numeric.
             continue
         if pd.isna(v): continue
         if np.isscalar(v) and hasattr(v, 'item'): pos_recent_dict[f"pos_recent_{c}"] = v.item()
         else: pos_recent_dict[f"pos_recent_{c}"] = v

    # NAME_CONTRACT_STATUS_COUNT
    status_counts = pos['NAME_CONTRACT_STATUS'].value_counts().to_dict()
    pos_recent_dict.update({f'pos_NAME_CONTRACT_STATUS_CNT_{k}': v for k, v in status_counts.items()})

    # aggregate features
    pos['YEAR_SCALE'] = (pos['MONTHS_BALANCE']/12.0).apply(np.exp)
    pos['SK_DPD_SCALE'] = pos['SK_DPD'] * pos['YEAR_SCALE']
    pos['SK_DPD_DEF_SCALE'] = pos['SK_DPD_DEF'] * pos['YEAR_SCALE']

    res = {**pos_recent_dict}
    
    # max
    for c in ['SK_DPD','SK_DPD_DEF']:
        v = pos[c].max()
        if not pd.isna(v): res[f"pos_max_{c}"] = v.item()

    # mean
    for c in ['SK_DPD','SK_DPD_DEF']:
        v = pos[c].mean()
        if not pd.isna(v): res[f"pos_mean_{c}"] = v.item()

    # sum / scaled mean
    pos_year_sum = pos['YEAR_SCALE'].sum()
    for c in ['SK_DPD_SCALE','SK_DPD_DEF_SCALE']:
        sum_v = pos[c].sum()
        if not pd.isna(sum_v):
            res[f"pos_sum_{c}"] = sum_v.item()
            if pos_year_sum != 0 and not pd.isna(pos_year_sum):
                res[f"pos_mean_{c}"] = (sum_v / pos_year_sum).item()

    # last DPD
    pos_dpd = pos[pos['SK_DPD'] > 0]
    if not pos_dpd.empty:
        v = pos_dpd['MONTHS_BALANCE'].max()
        if not pd.isna(v): res['pos_MONTH_LAST_DPD'] = v.item()

    res['pos_MONTH_CNT'] = len(pos)
    res['pos_MONTH_MAX'] = pos['MONTHS_BALANCE'].min().item()
    res['pos_count'] = pos['SK_ID_PREV'].nunique()
    
    return res

if __name__ == "__main__":
    import json
    data_dir = 'c:/Users/Admin/Desktop/MAS Swinburne/credi-council-swinhackathon/home-credit-default-risk'
    print(json.dumps(extract_pos_cash_features(100001, data_dir), indent=2))
