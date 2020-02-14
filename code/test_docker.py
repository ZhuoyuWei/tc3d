import xgboost

model_config = {'n_estimators': 300, 'max_depth': 8,
                'n_jobs': 16, 'tree_method': 'hist'}
lm_x = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                            max_depth=model_config['max_depth'],
                            n_jobs=model_config['n_jobs'],
                            random_state=42,
                            tree_method=model_config['tree_method'])

print('XGboost is successful')