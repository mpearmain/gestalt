# Wrappers API 

When running models from scikit-learn the interface to the model api is standardized making it easy to simply add a
compiled model to be run e.g.

```python
PARAMS_RF = {'n_estimators': 500,
             'criterion': 'gini',
             'n_jobs': 8,
             'verbose': 0,
             'random_state': 42,
             'oob_score': True,
             }

class ModelRF(BaseModel):
    def build_model(self):
        return RandomForestClassifier(**self.params)
```

Other models outside of the scikit-learn eco-system may have different api interfaces to call and run models
or (in the case of XGB and Keras) not all parameters are exposed in the sklearn interface they have made.

This is a set of common algorithms with wrappers to be able to run.

For each wrapper api the output call should be predict or predict_proba to keep in line with scikit-learn.


