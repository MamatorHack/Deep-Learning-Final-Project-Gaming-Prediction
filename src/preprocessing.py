from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

def prepare_all_tasks(df_reg, df_class):
    df_reg, df_class = df_reg.copy(), df_class.copy()
    
    # Régression
    X_reg = df_reg.drop(columns=['usd', 'Log_Hype']).values
    y_price, y_hype = df_reg['usd'].values, df_reg['Log_Hype'].values
    
    # Classification
    le_pop, le_country = LabelEncoder(), LabelEncoder()
    df_class['country_enc'] = le_country.fit_transform(df_class['country'])
    X_cls = df_class.drop(columns=['country', 'Popularity']).values
    y_cls = le_pop.fit_transform(df_class['Popularity'])

    Xr_tr, Xr_te, yp_tr, yp_te, yh_tr, yh_te = train_test_split(X_reg, y_price, y_hype, test_size=0.2, random_state=42)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42)

    sc_r, sc_c = RobustScaler(), RobustScaler()
    return (sc_r.fit_transform(Xr_tr), sc_r.transform(Xr_te), yp_tr, yp_te, yh_tr, yh_te, sc_r), \
           (sc_c.fit_transform(Xc_tr), sc_c.transform(Xc_te), yc_tr, yc_te, sc_c, le_pop, le_country)