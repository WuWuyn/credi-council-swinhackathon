import pandas as pd
import numpy as np
import os

def extract_application_features(sk_id_curr, data_dir):
    """
    Extracts engineered features derived from the main application_train.csv
    for a single SK_ID_CURR.
    """
    app_path = os.path.join(data_dir, 'application_train.csv')
    if not os.path.exists(app_path):
        return {}
        
    # In a real environment we might want to pre-load this or index it for efficiency,
    # but for individual tool calls we load and filter.
    df = pd.read_csv(app_path)
    
    # Needs some global aggregates for relative features
    # NOTE: Computing global medians on the fly for every call is extremely inefficient.
    # In a production MASCA pipeline, we'd cache these. For this hackathon demo, 
    # we'll compute them if we really need to, or omit them if they aren't strictly required
    # by LightGBM. The notebook uses them so we must compute them.
    
    # We first process the single row
    row = df[df['SK_ID_CURR'] == sk_id_curr].copy()
    if row.empty:
        return {}
        
    combined = row.copy()
    
    combined['CODE_GENDER'] = combined['CODE_GENDER'].replace('XNA', np.nan)
    combined['NAME_FAMILY_STATUS'] = combined['NAME_FAMILY_STATUS'].replace('Unknown', np.nan)
    combined['ORGANIZATION_TYPE'] = combined['ORGANIZATION_TYPE'].replace('XNA', np.nan)
    combined['DAYS_EMPLOYED'] = combined['DAYS_EMPLOYED'].where(combined['DAYS_EMPLOYED'] != 365243, np.nan)
    
    # document / live flags - only numeric
    docs = [f_ for f_ in combined.columns if 'FLAG_DOC' in f_ and combined[f_].dtype in ['int64','int32','float64','float32']]
    live = [f_ for f_ in combined.columns if ('FLAG_' in f_) and ('FLAG_DOC' not in f_) and ('_FLAG_' not in f_) and combined[f_].dtype in ['int64','int32','float64','float32']]
    combined['NEW_DOC_IND_KURT'] = combined[docs].kurtosis(axis=1) if docs else 0
    combined['NEW_LIVE_IND_SUM'] = combined[live].sum(axis=1) if live else 0
    
    combined['NEW_INC_PER_CHLD'] = combined['AMT_INCOME_TOTAL'] / (1 + combined['CNT_CHILDREN'])
    
    # Optional: global median mapping
    # inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    # If we don't map it globally, the model might fail. We will map via the global DF
    inc_by_org = df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].median()
    combined['NEW_INC_BY_ORG'] = combined['ORGANIZATION_TYPE'].map(inc_by_org)
    
    combined['NEW_EMPLOY_TO_BIRTH_RATIO'] = combined['DAYS_EMPLOYED'] / combined['DAYS_BIRTH']
    combined['NEW_SOURCES_PROD'] = combined['EXT_SOURCE_1'] * combined['EXT_SOURCE_2'] * combined['EXT_SOURCE_3']
    combined['NEW_EXT_SOURCES_MEAN'] = combined[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    combined['NEW_SCORES_STD'] = combined[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    # The notebook filled with mean, we'll leave as NA if it's 1 row and std is NaN (which it is for 1 row, wait std of 3 cols)
    
    combined['NEW_CAR_TO_BIRTH_RATIO'] = combined['OWN_CAR_AGE'] / combined['DAYS_BIRTH']
    combined['NEW_CAR_TO_EMPLOY_RATIO'] = combined['OWN_CAR_AGE'] / combined['DAYS_EMPLOYED']
    combined['NEW_PHONE_TO_BIRTH_RATIO'] = combined['DAYS_LAST_PHONE_CHANGE'] / combined['DAYS_BIRTH']
    combined['NEW_PHONE_TO_EMPLOYED_RATIO'] = combined['DAYS_LAST_PHONE_CHANGE'] / combined['DAYS_EMPLOYED']
    combined['NEW_CREDIT_TO_INCOME_RATIO'] = combined['AMT_CREDIT'] / combined['AMT_INCOME_TOTAL'].replace(0, np.nan)
    
    combined['AMT_PAY_YEAR'] = combined['AMT_CREDIT'] / combined['AMT_ANNUITY'].replace(0, np.nan)
    combined['AGE_PAYOFF'] = -combined['DAYS_BIRTH']/365.25 + combined['AMT_PAY_YEAR']
    combined['AMT_ANNUITY_INCOME_RATE'] = combined['AMT_ANNUITY'] / combined['AMT_INCOME_TOTAL'].replace(0, np.nan)
    combined['AMT_DIFF_CREDIT_GOODS'] = combined['AMT_CREDIT'] - combined['AMT_GOODS_PRICE']
    combined['AMT_CREDIT_GOODS_PERC'] = combined['AMT_CREDIT'] / combined['AMT_GOODS_PRICE'].replace(0, np.nan)
    
    doc_cols = [c for c in combined.columns if c.startswith('FLAG_DOCUMENT') and combined[c].dtype in ['int64','int32','float64','float32']]
    combined['DOCUMENT_CNT'] = combined[doc_cols].sum(axis=1) if doc_cols else 0
    combined['AGE_EMPLOYED'] = combined['DAYS_EMPLOYED'] - combined['DAYS_BIRTH']
    combined['AMT_INCOME_OVER_CHILD'] = combined['AMT_INCOME_TOTAL'] / combined['CNT_CHILDREN'].replace(0, np.nan)
    combined['CNT_ADULT'] = combined['CNT_FAM_MEMBERS'] - combined['CNT_CHILDREN']
    combined['ADULT_RATIO'] = combined['CNT_ADULT'] / combined['CNT_FAM_MEMBERS'].replace(0, np.nan)
    
    combined['AMT_REQ_CREDIT_BUREAU_MON_CHANGE'] = combined['AMT_REQ_CREDIT_BUREAU_QRT']/2 - combined['AMT_REQ_CREDIT_BUREAU_MON']
    combined['AMT_REQ_CREDIT_BUREAU_QRT_CHANGE'] = combined['AMT_REQ_CREDIT_BUREAU_YEAR']/3 - combined['AMT_REQ_CREDIT_BUREAU_QRT']
    combined['AMT_REQ_CREDIT_BUREAU_TOTAL'] = combined['AMT_REQ_CREDIT_BUREAU_HOUR'] + combined['AMT_REQ_CREDIT_BUREAU_DAY'] \
                                            + combined['AMT_REQ_CREDIT_BUREAU_MON'] + combined['AMT_REQ_CREDIT_BUREAU_QRT'] \
                                            + combined['AMT_REQ_CREDIT_BUREAU_YEAR']
                                            
    # Factorize REGION
    # Region factorize is tricky via single row. We skip it, LightGBM might handle NaNs.
    
    df['CNT_CHILDREN_CLIPPED'] = df['CNT_CHILDREN'].clip(0, 10)
    df['GENDER_FAMILY_STATUS'] = df['CODE_GENDER'].astype(str) + df['NAME_FAMILY_STATUS'].astype(str)
    combined['CNT_CHILDREN_CLIPPED'] = combined['CNT_CHILDREN'].clip(0, 10)
    combined['GENDER_FAMILY_STATUS'] = combined['CODE_GENDER'].astype(str) + combined['NAME_FAMILY_STATUS'].astype(str)
    
    combined['gender_mean_income'] = combined['CODE_GENDER'].map(df.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].median())
    combined['own_car_mean_income'] = combined['FLAG_OWN_CAR'].map(df.groupby('FLAG_OWN_CAR')['AMT_INCOME_TOTAL'].median())
    combined['own_realty_mean_income'] = combined['FLAG_OWN_REALTY'].map(df.groupby('FLAG_OWN_REALTY')['AMT_INCOME_TOTAL'].median())
    combined['cnt_children_mean_income'] = combined['CNT_CHILDREN_CLIPPED'].map(df.groupby('CNT_CHILDREN_CLIPPED')['AMT_INCOME_TOTAL'].median())
    # combined['region_mean_income'] = combined['REGION'].map(...) skipping region
    combined['family_status_mean_income'] = combined['NAME_FAMILY_STATUS'].map(df.groupby('NAME_FAMILY_STATUS')['AMT_INCOME_TOTAL'].median())
    combined['gender_family_status_mean_income'] = combined['GENDER_FAMILY_STATUS'].map(df.groupby('GENDER_FAMILY_STATUS')['AMT_INCOME_TOTAL'].median())
    
    combined['gender_mean_income_rel'] = (combined['AMT_INCOME_TOTAL'] - combined['gender_mean_income']) / combined['gender_mean_income']
    combined['own_car_mean_income_rel'] = (combined['AMT_INCOME_TOTAL'] - combined['own_car_mean_income']) / combined['own_car_mean_income']
    combined['own_realty_mean_income_rel'] = (combined['AMT_INCOME_TOTAL'] - combined['own_realty_mean_income']) / combined['own_realty_mean_income']
    combined['cnt_children_mean_income_rel'] = (combined['AMT_INCOME_TOTAL'] - combined['cnt_children_mean_income']) / combined['cnt_children_mean_income']
    combined['family_status_mean_income_rel'] = (combined['AMT_INCOME_TOTAL'] - combined['family_status_mean_income']) / combined['family_status_mean_income']
    combined['gender_family_status_mean_income_rel'] = (combined['AMT_INCOME_TOTAL'] - combined['gender_family_status_mean_income']) / combined['gender_family_status_mean_income']

    rejected_features = ['AMT_GOODS_PRICE','APARTMENTS_AVG','APARTMENTS_MEDI',
                         'BASEMENTAREA_AVG','BASEMENTAREA_MODE','COMMONAREA_AVG','COMMONAREA_MODE',
                         'ELEVATORS_AVG','ELEVATORS_MEDI','ENTRANCES_AVG','ENTRANCES_MEDI','FLOORSMAX_AVG','FLOORSMAX_MEDI',
                         'FLOORSMIN_AVG','FLOORSMIN_MEDI','LANDAREA_AVG','LANDAREA_MODE',
                         'LIVINGAPARTMENTS_AVG','LIVINGAPARTMENTS_MEDI',
                         'LIVINGAREA_AVG','LIVINGAREA_MODE',
                         'NONLIVINGAPARTMENTS_AVG','NONLIVINGAPARTMENTS_MEDI',
                         'NONLIVINGAREA_AVG','NONLIVINGAREA_MODE','OBS_30_CNT_SOCIAL_CIRCLE',
                         'REGION_RATING_CLIENT','YEARS_BEGINEXPLUATATION_AVG','YEARS_BEGINEXPLUATATION_MEDI',
                         'YEARS_BUILD_AVG','YEARS_BUILD_MEDI']
    rejected_features += ['ELEVATORS_MODE','ENTRANCE_MODE','FLOORSMAX_MEDI','FLOORSMIN_MEDI','NONLIVINGAPARTMENTS_MODE']
    rejected_features += ['FLAG_MOBIL','FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_2','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START']
    rejected_features += ['gender_mean_income', 'own_car_mean_income', 'own_realty_mean_income', 'cnt_children_mean_income', 'family_status_mean_income', 'gender_family_status_mean_income', 'CNT_CHILDREN_CLIPPED']
    
    for f_ in rejected_features:
        if f_ in combined.columns:
            del combined[f_]
            
    # Remove categorical columns that would normally be factorized
    # (Since we are scoring a single instance, LightGBM might be able to handle NaNs if it expects categorical codes)
    cat_feats = [f for f in combined.columns if combined[f].dtype == 'object']
    for f in cat_feats:
         del combined[f]
         
    res = {}
    for c in combined.columns:
        if c == 'TARGET': continue
        v = combined[c].values[0]
        if pd.isna(v): continue
        if np.isscalar(v) and hasattr(v, 'item'):
            res[c] = v.item()
        elif isinstance(v, (int, float)):
            res[c] = v
            
    return res

if __name__ == "__main__":
    import json
    data_dir = 'c:/Users/Admin/Desktop/MAS Swinburne/credi-council-swinhackathon/home-credit-default-risk'
    print(json.dumps(extract_application_features(100002, data_dir), indent=2))
