import pandas as pd
import numpy as np
import os

def extract_bureau_features(sk_id_curr, data_dir):
    """
    Extracts bureau features for a single SK_ID_CURR, replicating the logic from lgb1.ipynb.
    """
    bureau_path = os.path.join(data_dir, 'bureau.csv')
    bubl_path = os.path.join(data_dir, 'bureau_balance.csv')
    
    if not os.path.exists(bureau_path) or not os.path.exists(bubl_path):
        return {} # Return empty dict if data missing

    # Load and filter bureau data for this applicant
    buro = pd.read_csv(bureau_path)
    buro = buro[buro['SK_ID_CURR'] == sk_id_curr].copy()
    
    if buro.empty:
        return {}
    
    # Load and filter bubl data for the associated SK_ID_BUREAU
    bubl = pd.read_csv(bubl_path)
    bubl = bubl[bubl['SK_ID_BUREAU'].isin(buro['SK_ID_BUREAU'])].copy()
    
    # --- BUBL Features (From Cell 5) ---
    if not bubl.empty:
        bubl_last_DPD = bubl[bubl['STATUS'].isin(['1','2','3','4','5'])].groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].max()
        bubl_last_DPD.rename('MONTH_LAST_DPD', inplace=True)

        bubl_last_C = bubl[bubl['STATUS'] == 'C'].groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].min()
        bubl_last_C.rename('MONTH_LAST_C', inplace=True)

        STATUS_TCNT = pd.Series(bubl.groupby('SK_ID_BUREAU')['STATUS'].value_counts()).rename('STATUS_TCNT')
        STATUS_TCNT = pd.pivot_table(STATUS_TCNT.reset_index(), index='SK_ID_BUREAU', columns='STATUS', values='STATUS_TCNT', fill_value=0)
        
        # Ensure all columns 0-5 exist for summing
        for i in range(6):
            if str(i) not in STATUS_TCNT.columns:
                STATUS_TCNT[str(i)] = 0
                
        STATUS_TCNT['DPD_SUM'] = np.zeros([STATUS_TCNT.shape[0]])
        count = np.zeros([STATUS_TCNT.shape[0]])
        for i in range(0,6):
            STATUS_TCNT['DPD_SUM'] += STATUS_TCNT[str(i)]*i
            count += STATUS_TCNT[str(i)]
            del STATUS_TCNT[str(i)]
        STATUS_TCNT['DPD_MEAN'] = STATUS_TCNT['DPD_SUM']/(count+0.0001)
        STATUS_TCNT.columns = ['STATUS_TCNT_' + f_ for f_ in STATUS_TCNT.columns]

        #over recent 12 months
        bubl_12m = bubl[bubl['MONTHS_BALANCE']>=-12]
        if not bubl_12m.empty:
            STATUS_12CNT = pd.Series(bubl_12m.groupby('SK_ID_BUREAU')['STATUS'].value_counts()).rename('STATUS_6CNT')  
            STATUS_12CNT = pd.pivot_table(STATUS_12CNT.reset_index(), index='SK_ID_BUREAU',columns='STATUS',values='STATUS_6CNT',fill_value=0)
            
            for i in range(6):
                if str(i) not in STATUS_12CNT.columns:
                    STATUS_12CNT[str(i)] = 0
            
            STATUS_12CNT['DPD_SUM'] = np.zeros([STATUS_12CNT.shape[0]])
            count = np.zeros([STATUS_12CNT.shape[0]])
            for i in range(0,6):
                STATUS_12CNT['DPD_SUM'] += STATUS_12CNT[str(i)]*i
                count += STATUS_12CNT[str(i)]
                del STATUS_12CNT[str(i)]
            STATUS_12CNT['DPD_MEAN'] = STATUS_12CNT['DPD_SUM']/(count+0.0001)
            STATUS_12CNT.columns = ['STATUS_12CNT_' + f_ for f_ in STATUS_12CNT.columns]
        else:
            STATUS_12CNT = pd.DataFrame(index=bubl['SK_ID_BUREAU'].unique())
            STATUS_12CNT['STATUS_12CNT_DPD_SUM'] = 0
            STATUS_12CNT['STATUS_12CNT_DPD_MEAN'] = 0
            STATUS_12CNT['STATUS_12CNT_C'] = 0
            STATUS_12CNT['STATUS_12CNT_X'] = 0

        # Merge bubl features back to buro
        buro = buro.merge(bubl_last_DPD, on='SK_ID_BUREAU', how='left')
        buro = buro.merge(bubl_last_C, on='SK_ID_BUREAU', how='left')
        buro = buro.merge(STATUS_TCNT, on='SK_ID_BUREAU', how='left')
        buro = buro.merge(STATUS_12CNT, on='SK_ID_BUREAU', how='left')
    else:
        # Fill missing bubl columns with NaNs or 0s
        for c in ['MONTH_LAST_DPD', 'MONTH_LAST_C', 'STATUS_TCNT_C', 'STATUS_TCNT_X', 'STATUS_TCNT_DPD_SUM', 'STATUS_TCNT_DPD_MEAN',
                  'STATUS_12CNT_C', 'STATUS_12CNT_X', 'STATUS_12CNT_DPD_SUM', 'STATUS_12CNT_DPD_MEAN']:
            buro[c] = np.nan

    # --- Bureau Features (From Cell 6) ---
    buro.loc[buro['DAYS_CREDIT_ENDDATE'] < -40000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    buro.loc[buro['DAYS_CREDIT_UPDATE'] < -40000, 'DAYS_CREDIT_UPDATE'] = np.nan
    buro.loc[buro['DAYS_ENDDATE_FACT'] < -40000, 'DAYS_ENDDATE_FACT'] = np.nan

    buro['AMT_DEBT_RATIO'] = buro['AMT_CREDIT_SUM_DEBT']/(1+buro['AMT_CREDIT_SUM'])
    buro['AMT_LIMIT_RATIO'] = buro['AMT_CREDIT_SUM_LIMIT']/(1+buro['AMT_CREDIT_SUM'])
    buro['AMT_SUM_OVERDUE_RATIO'] = buro['AMT_CREDIT_SUM_OVERDUE']/(1+buro['AMT_CREDIT_SUM'])
    buro['AMT_MAX_OVERDUE_RATIO'] = buro['AMT_CREDIT_MAX_OVERDUE']/(1+buro['AMT_CREDIT_SUM'])
    buro['DAYS_END_DIFF'] = buro['DAYS_ENDDATE_FACT'] - buro['DAYS_CREDIT_ENDDATE']

    buro = pd.concat([buro, pd.get_dummies(buro.CREDIT_ACTIVE, prefix='CREDIT_ACTIVE')], axis=1)
    buro = pd.concat([buro, pd.get_dummies(buro.CREDIT_CURRENCY, prefix='CREDIT_CURRENCY')], axis=1)
    buro = pd.concat([buro, pd.get_dummies(buro.CREDIT_TYPE, prefix='CREDIT_TYPE')], axis=1)
    
    # Active Bureau logic
    active_buro = buro.loc[buro['CREDIT_ACTIVE_Active'] == 1].copy()
    if not active_buro.empty:
        active_buro['DAYS_LEFT_RATIO'] = active_buro['DAYS_CREDIT_ENDDATE']/(active_buro['DAYS_CREDIT_ENDDATE']-active_buro['DAYS_CREDIT'])
        active_buro['AMT_CREDIT_LEFT'] = active_buro['AMT_CREDIT_SUM'] * active_buro['DAYS_LEFT_RATIO']
        active_buro['AMT_CREDIT_LEFT_OVER_ANNUITY'] = active_buro['AMT_CREDIT_LEFT'] / active_buro['AMT_ANNUITY']
    
    # Aggregation
    avg_feats = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE', 'DAYS_END_DIFF']
    sum_feats = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY', 'AMT_DEBT_RATIO', 'AMT_LIMIT_RATIO', 'AMT_SUM_OVERDUE_RATIO', 'AMT_MAX_OVERDUE_RATIO', 'DAYS_END_DIFF', 'STATUS_TCNT_C', 'STATUS_TCNT_X', 'STATUS_TCNT_DPD_SUM', 'STATUS_TCNT_DPD_MEAN', 'STATUS_12CNT_C', 'STATUS_12CNT_X', 'STATUS_12CNT_DPD_SUM', 'STATUS_12CNT_DPD_MEAN', 'MONTH_LAST_DPD', 'MONTH_LAST_C']
    sum_feats += [c for c in buro.columns if c.startswith('CREDIT_ACTIVE_') or c.startswith('CREDIT_CURRENCY_') or c.startswith('CREDIT_TYPE_')]
    min_feats = ['MONTH_LAST_DPD', 'MONTH_LAST_C', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE']
    max_feats = ['MONTH_LAST_DPD', 'MONTH_LAST_C', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE']

    # We are calculating for a single SK_ID_CURR, so we extract the single-row dict
    res = {}
    
    # General Buro Aggr
    for f in avg_feats:
        if f in buro.columns and not pd.isna(buro[f].mean()):
            res[f"bureau_avg_{f}"] = buro[f].mean().item()
    for f in sum_feats:
        if f in buro.columns and not pd.isna(buro[f].sum()):
            res[f"bureau_sum_{f}"] = buro[f].sum().item()
    for f in min_feats:
        if f in buro.columns and not pd.isna(buro[f].min()):
            res[f"bureau_min_{f}"] = buro[f].min().item()
    for f in max_feats:
        if f in buro.columns and not pd.isna(buro[f].max()):
            res[f"bureau_max_{f}"] = buro[f].max().item()
            
    # Active Buro Aggr
    active_avg_feats = avg_feats + ['AMT_CREDIT_LEFT', 'AMT_CREDIT_LEFT_OVER_ANNUITY', 'DAYS_LEFT_RATIO']
    active_sum_feats = sum_feats + ['AMT_CREDIT_LEFT', 'AMT_CREDIT_LEFT_OVER_ANNUITY']
    if not active_buro.empty:
        for f in active_avg_feats:
            if f in active_buro.columns and not pd.isna(active_buro[f].mean()):
                res[f"active_avg_{f}"] = active_buro[f].mean().item()
        for f in active_sum_feats:
            if f in active_buro.columns and not pd.isna(active_buro[f].sum()):
                res[f"active_sum_{f}"] = active_buro[f].sum().item()
        
    return res

if __name__ == "__main__":
    # Test script
    import json
    data_dir = 'c:/Users/Admin/Desktop/MAS Swinburne/home-credit-default-risk-master/data'
    # Fallback to the real data folder we mapped
    data_dir = 'c:/Users/Admin/Desktop/MAS Swinburne/credi-council-swinhackathon/home-credit-default-risk'
    print(json.dumps(extract_bureau_features(100001, data_dir), indent=2))
