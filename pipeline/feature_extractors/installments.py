import pandas as pd
import numpy as np
import os

def extract_installments_features(sk_id_curr, data_dir):
    """
    Extracts features derived from installments_payments for a single SK_ID_CURR.
    """
    inst_path = os.path.join(data_dir, 'installments_payments.csv')
    if not os.path.exists(inst_path):
        return {}

    # Load and filter Data
    inst = pd.read_csv(inst_path)
    inst = inst[inst['SK_ID_CURR'] == sk_id_curr].copy()
    if inst.empty:
        return {}
        
    inst_NUM_INSTALMENT_VERSION = inst['NUM_INSTALMENT_VERSION'].nunique()
    
    inst['DAYS_ENTRY_PAYMENT_weighted'] = inst['DAYS_ENTRY_PAYMENT'] * inst['AMT_PAYMENT']
    inst = inst.groupby(['SK_ID_PREV','SK_ID_CURR','NUM_INSTALMENT_NUMBER']).agg({
        'DAYS_INSTALMENT':'mean',
        'DAYS_ENTRY_PAYMENT_weighted':'sum',
        'AMT_INSTALMENT':'mean',
        'AMT_PAYMENT':'sum'
    })
    
    inst['DAYS_ENTRY_PAYMENT'] = inst['DAYS_ENTRY_PAYMENT_weighted'] / inst['AMT_PAYMENT'].replace(0, np.nan)
    inst = inst.reset_index()
    del inst['DAYS_ENTRY_PAYMENT_weighted']
    
    # Calculate features
    inst['AMT_PAYMENT_PERC'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT'].replace(0, np.nan)
    inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['DBD'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']
    inst['DPD'] = inst['DPD'].apply(lambda x: x if x > 0 else 0)
    inst['DBD'] = inst['DBD'].apply(lambda x: x if x > 0 else 0)
    inst['DPD'] = inst['DPD'].fillna(30)
    inst['DBD'] = inst['DBD'].fillna(0)
    
    inst['AMT_PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['DAYS_ENTRY_PAYMENT_SCALE'] = (inst['DAYS_ENTRY_PAYMENT']/365.25).apply(np.exp)
    inst['DPD_SCALE'] = inst['DPD'] * inst['DAYS_ENTRY_PAYMENT_SCALE']
    inst['DBD_SCALE'] = inst['DBD'] * inst['DAYS_ENTRY_PAYMENT_SCALE']
    inst['AMT_PAYMENT_DIFF_SCALE'] = inst['AMT_PAYMENT_DIFF'] * inst['DAYS_ENTRY_PAYMENT_SCALE']
    inst['AMT_PAYMENT_SCALE'] = inst['AMT_PAYMENT'] * inst['DAYS_ENTRY_PAYMENT_SCALE']
    
    res = {}
    
    # max
    for c in ['DPD','DBD','AMT_PAYMENT_DIFF','AMT_PAYMENT_PERC']:
        v = inst[c].max()
        if not pd.isna(v): res[f"inst_max_{c}"] = v.item()
        
    # var
    for c in ['DPD','DBD','AMT_PAYMENT_DIFF','AMT_PAYMENT_PERC']:
        v = inst[c].var()
        if not pd.isna(v): res[f"inst_var_{c}"] = v.item()

    # sum / scaled mean
    inst_day_scale_sum = inst['DAYS_ENTRY_PAYMENT_SCALE'].sum()
    for c in ['DPD_SCALE','DBD_SCALE','AMT_PAYMENT_DIFF_SCALE','AMT_PAYMENT_SCALE']:
        sum_v = inst[c].sum()
        if not pd.isna(sum_v):
            res[f"inst_sum_{c}"] = sum_v.item()
            if inst_day_scale_sum != 0 and not pd.isna(inst_day_scale_sum):
                res[f"inst_mean_{c}"] = (sum_v / inst_day_scale_sum).item()
                
    # avg
    for c in ['DPD','DBD','AMT_PAYMENT_DIFF','AMT_PAYMENT','AMT_PAYMENT_PERC']:
        v = inst[c].mean()
        if not pd.isna(v): res[f"inst_mean_{c}"] = v.item()
        
    # last late
    inst_late = inst[inst['DAYS_INSTALMENT'] < inst['DAYS_ENTRY_PAYMENT']]
    if not inst_late.empty:
        v = inst_late['DAYS_INSTALMENT'].max()
        if not pd.isna(v): res['inst_DAYS_LAST_LATE'] = v.item()
        
    # last underpaid
    inst_underpaid = inst[inst['AMT_INSTALMENT'] < inst['AMT_PAYMENT']]
    if not inst_underpaid.empty:
        v = inst_underpaid['DAYS_INSTALMENT'].max()
        if not pd.isna(v): res['inst_DAYS_LAST_UNDERPAID'] = v.item()
        
    res['inst_N_NUM_INSTALMENT_VERSION'] = inst_NUM_INSTALMENT_VERSION.item() if hasattr(inst_NUM_INSTALMENT_VERSION, 'item') else inst_NUM_INSTALMENT_VERSION
    
    amt_inst_sum = inst['AMT_INSTALMENT'].sum()
    if amt_inst_sum != 0:
        res['inst_AMT_PAYMENT_TOTAL_RATIO'] = (inst['AMT_PAYMENT'].sum() / amt_inst_sum).item()
        
    res['inst_length'] = len(inst)
    res['inst_count'] = inst['SK_ID_PREV'].nunique()
    
    return res

if __name__ == "__main__":
    import json
    data_dir = 'c:/Users/Admin/Desktop/MAS Swinburne/credi-council-swinhackathon/home-credit-default-risk'
    print(json.dumps(extract_installments_features(100001, data_dir), indent=2))
