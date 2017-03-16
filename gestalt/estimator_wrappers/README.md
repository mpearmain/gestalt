# Wrappers API 

When running models from scikit-learn the interface to the model api is standardized making it easy to simply add a
compiled model to be run e.g.

Other models outside of the scikit-learn eco-system may have different api interfaces to call and run models
or (in the case of XGB and Keras) not all parameters are exposed in the sklearn interface they have made.

This is a set of common algorithms with wrappers to be able to run.

For each wrapper api the output calls should be `fit`, `predict` or `predict_proba` to keep in line with scikit-learn.


