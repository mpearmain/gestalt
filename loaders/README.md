# Loaders API 

When building the datasets to run our algorithms against data can either com in the form of pandas dataframes or libsvm
style data.

We provide three basic loaders:
1. A loader when all data can be read as dense pandas DataFrames
2. A loader when all data is can be read as libsvm 
3. A loader for a mix of dense and sparse (the resultant data is sparse)

Under the hood we use scikit-learn utility functions for loading datasets in the svmlight / libsvm format. 
In this format, each line takes the form <label> <feature-id>:<feature-value> <feature-id>:<feature-value> .... 
This format is especially suitable for sparse datasets. 

Scipy sparse CSR matrices are used for X and numpy arrays are used for y.


