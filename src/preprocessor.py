from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


#Created numeric and categorical pipeline.
# Filled missing with median/'missing'. 
# Scaled/One Hot Encoded
def get_preprocessor():

    numeric_features = ['age', 'myear']
    categorical_features = ['peakid', 'mseason', 'sex', 'citizen', 'status', 'residence', 'occupation', 
                        'leader','deputy', 'bconly', 'nottobc', 'support', 'disabled', 'hired', 'sherpa',
                        'tibetan', 'msolo', 'mtraverse', 'mski', 'mparapente', 'mspeed', 'mo2climb', 'mo2sleep',
                        'mo2medical', 'mo2note', 'route_choice', 'oxygen_status']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())    
    ])


    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])

    return ColumnTransformer( transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

