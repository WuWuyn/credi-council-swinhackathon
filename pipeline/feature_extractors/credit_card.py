import pandas as pd
import numpy as np
import os

def extract_credit_card_features(sk_id_curr, data_dir):
    """
    Extracts features derived from credit_card_balance for a single SK_ID_CURR.
    """
    ccbl_path = os.path.join(data_dir, 'credit_card_balance.csv')
    if not os.path.exists(ccbl_path):
        return {}

    ccbl = pd.read_csv(ccbl_path)
    ccbl = ccbl[ccbl['SK_ID_CURR'] == sk_id_curr].copy()
    if ccbl.empty:
        return {}
        
    sum_feats = [f_ for f_ in ccbl.columns.values if (('AMT' in f_) or ('SK_DPD' in f_) or ('CNT' in f_)) and ('CUM' not in f_)]
    
    # We do a month grouping. For a single SK_ID_CURR, this is just grouping by MONTHS_BALANCE
    sum_ccbl_mon = ccbl.groupby(['MONTHS_BALANCE'])[sum_feats].sum()
    sum_ccbl_mon['CNT_ACCOUNT_W_MONTH'] = ccbl.groupby(['MONTHS_BALANCE'])['SK_ID_PREV'].count()
    sum_ccbl_mon = sum_ccbl_mon.reset_index()
    
    # Ratios
    sum_ccbl_mon['AMT_BALANCE_CREDIT_RATIO'] = (sum_ccbl_mon['AMT_BALANCE']/(sum_ccbl_mon['AMT_CREDIT_LIMIT_ACTUAL']+0.001)).clip(-100,100)
    sum_ccbl_mon['AMT_CREDIT_USE_RATIO'] = (sum_ccbl_mon['AMT_DRAWINGS_CURRENT']/(sum_ccbl_mon['AMT_CREDIT_LIMIT_ACTUAL']+0.001)).clip(-100,100)
    sum_ccbl_mon['AMT_DRAWING_ATM_RATIO'] = sum_ccbl_mon['AMT_DRAWINGS_ATM_CURRENT']/(sum_ccbl_mon['AMT_DRAWINGS_CURRENT']+0.001)
    if 'AMT_DRAWINGS_OTHER_CURRENT' in sum_ccbl_mon.columns:
         sum_ccbl_mon['AMT_DRAWINGS_OTHER_RATIO'] = sum_ccbl_mon['AMT_DRAWINGS_OTHER_CURRENT']/(sum_ccbl_mon['AMT_DRAWINGS_CURRENT']+0.001)
    if 'AMT_DRAWINGS_POS_CURRENT' in sum_ccbl_mon.columns:
         sum_ccbl_mon['AMT_DRAWINGS_POS_RATIO'] = sum_ccbl_mon['AMT_DRAWINGS_POS_CURRENT']/(sum_ccbl_mon['AMT_DRAWINGS_CURRENT']+0.001)
    sum_ccbl_mon['AMT_PAY_USE_RATIO'] = ((sum_ccbl_mon['AMT_PAYMENT_TOTAL_CURRENT']+0.001)/(sum_ccbl_mon['AMT_DRAWINGS_CURRENT']+0.001)).clip(-100,100)
    sum_ccbl_mon['AMT_BALANCE_RECIVABLE_RATIO'] = sum_ccbl_mon['AMT_BALANCE']/(sum_ccbl_mon['AMT_TOTAL_RECEIVABLE']+0.001)
    sum_ccbl_mon['AMT_DRAWING_BALANCE_RATIO'] = sum_ccbl_mon['AMT_DRAWINGS_CURRENT']/(sum_ccbl_mon['AMT_BALANCE']+0.001)
    sum_ccbl_mon['AMT_RECEIVABLE_PRINCIPAL_DIFF'] = sum_ccbl_mon['AMT_TOTAL_RECEIVABLE']-sum_ccbl_mon['AMT_RECEIVABLE_PRINCIPAL']
    sum_ccbl_mon['AMT_PAY_INST_DIFF'] = sum_ccbl_mon['AMT_PAYMENT_CURRENT'] - sum_ccbl_mon['AMT_INST_MIN_REGULARITY']
    
    rejected_features = ['AMT_RECIVABLE','AMT_RECEIVABLE_PRINCIPAL',
                         'AMT_DRAWINGS_OTHER_CURRENT','AMT_DRAWINGS_POS_CURRENT']
    for f_ in rejected_features:
        if f_ in sum_ccbl_mon.columns:
            del sum_ccbl_mon[f_]
            
    sum_feats = [f_ for f_ in sum_ccbl_mon.columns.values if (('AMT' in f_) or ('SK_DPD' in f_) or ('CNT' in f_)) and ('CUM' not in f_)]
    
    res = {}
    
    # mean over different windows
    def get_window_means(window, prefix):
        w_df = sum_ccbl_mon.loc[sum_ccbl_mon.MONTHS_BALANCE >= -window]
        if w_df.empty: return
        w_mean = w_df[sum_feats].mean()
        for k, v in w_mean.items():
            if not pd.isna(v): res[f"{prefix}_{k}"] = v.item()
            
    get_window_means(4, 'cc_mean4')
    get_window_means(12, 'cc_mean12')
    get_window_means(36, 'cc_mean36')
    
    # scale sum and mean
    sum_ccbl_mon2 = sum_ccbl_mon.copy()
    sum_ccbl_mon2['YEAR_SCALE'] = (sum_ccbl_mon2['MONTHS_BALANCE']/12.0).apply(np.exp)
    for f_ in sum_feats:
         sum_ccbl_mon2[f_] = sum_ccbl_mon2[f_] * sum_ccbl_mon2['YEAR_SCALE']
         
    scale_sum = sum_ccbl_mon2[sum_feats].sum()
    year_scale_sum = sum_ccbl_mon2['YEAR_SCALE'].sum()
    for k, v in scale_sum.items():
         if not pd.isna(v): 
             res[f"cc_scale_sum_{k}"] = v.item()
             if year_scale_sum != 0 and not pd.isna(year_scale_sum):
                 res[f"cc_scale_mean_{k}"] = (v / year_scale_sum).item()
                 
    # global mean, var, max, min
    for f_ in sum_feats:
        # mean
        v = sum_ccbl_mon[f_].mean()
        if not pd.isna(v): res[f"cc_mean_{f_}"] = v.item()
        # var
        v = sum_ccbl_mon[f_].var()
        if not pd.isna(v): res[f"cc_var_{f_}"] = v.item()
        # max
        v = sum_ccbl_mon[f_].max()
        if not pd.isna(v): res[f"cc_max_{f_}"] = v.item()

    if 'AMT_TOTAL_RECEIVABLE' in sum_ccbl_mon.columns:
         v = sum_ccbl_mon['AMT_TOTAL_RECEIVABLE'].min()
         if not pd.isna(v): res['cc_min_AMT_TOTAL_RECEIVABLE'] = v.item()
         
    if 'AMT_RECEIVABLE_PRINCIPAL_DIFF' in sum_ccbl_mon.columns:
         v = sum_ccbl_mon['AMT_RECEIVABLE_PRINCIPAL_DIFF'].min()
         if not pd.isna(v): res['cc_min_AMT_RECEIVABLE_PRINCIPAL_DIFF'] = v.item()
         
    # last DPD
    ccbl_dpd = ccbl[ccbl['SK_DPD'] > 0]
    if not ccbl_dpd.empty:
         v = ccbl_dpd['MONTHS_BALANCE'].max()
         if not pd.isna(v) and v != 0: res['cc_MONTH_LAST_DPD'] = v.item()
         
    ccbl_dpd7 = ccbl[ccbl['SK_DPD_DEF'] > 7]
    if not ccbl_dpd7.empty:
         v = ccbl_dpd7['MONTHS_BALANCE'].max()
         if not pd.isna(v) and v != 0: res['cc_MONTH_LAST_DPD7'] = v.item()
         
    # recent 
    idx = ccbl['MONTHS_BALANCE'].idxmax()
    if not pd.isna(idx):
        recent = ccbl.loc[[idx]]
        for c in ['MONTHS_BALANCE','CNT_INSTALMENT_MATURE_CUM','NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']:
            if c in recent.columns:
                v = recent[c].values[0]
                if c == 'NAME_CONTRACT_STATUS': 
                    # Skipped categorical factorize globally for single record approach like PosCash
                    continue
                if pd.isna(v): continue
                if np.isscalar(v) and hasattr(v, 'item'): res[f"cc_{c}"] = v.item()
                else: res[f"cc_{c}"] = v
                
    # Contract status counts over the history
    status_counts = ccbl['NAME_CONTRACT_STATUS'].value_counts(dropna=False).to_dict()
    for k, v in status_counts.items():
        if pd.isna(k): continue
        res[f"cc_{k}"] = v
        
    res['cc_history_len'] = len(ccbl)

    return res

if __name__ == "__main__":
    import json
    data_dir = 'c:/Users/Admin/Desktop/MAS Swinburne/credi-council-swinhackathon/home-credit-default-risk'
    print(json.dumps(extract_credit_card_features(100006, data_dir), indent=2))
