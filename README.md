# Gestalt

A helper library for data science pipelines

"_Something that is made of many parts and yet is somehow more than or different from the combination of its parts_"

Machine learning (ML) has achieved considerable success in multiple fields, much of the pipeline surrounding machine 
learning can be generalised to a point that removes the 'engineering' of the data flows and pipelines leaving the human
expert to work on the most difficult aspects:

1. Pre-processing the data
2. Feature Engineering
3. Select appropriate features

## The goals of Gestalt
The goal of Gestalt is to remove the cumbersome parts of building a meta-learner data science pipeline (i.e the process
of building a generalised stacker across folds).

The current roadmap for this module is as follows:

1. Build P.O.C for GeneralisedStacking class using pandas - Done
2. Build Stackers for, Regression and Classification problems (binary and multiclass) - Done
3. Create plugin wrapper to expose other algorithms to the stacking process (say R libraries) - Done
4. Support `scipy.sparse` data to allow both dense and sparse models to be run on the same folds -Done
5. Create an example of bayesian encoding as a transformer that runs across folds in line with the stacker.
6. Create Hyper-parameter autotuning class for the set of base models to be used in the metaleaner
7. Add loads of tests and documentation.

As with all alpha OSS projects things are under constant development and already one can see places where refactoring some
of the core code would make sense (i.e a Class to remove the cumbersome `_predict_#` and `_fit_#` methods)


## Install 
```python

pip install -U git+https://github.com/mpearmain/gestalt

_____
NOTE:
    gestalt is built using python 3.5
```

## Example 
In the examples dir there is a set of examples showing various different use cases.
To run the r wrapper for ranger you need to have a  copy of `rpy2` and a copy of the `ranger` R library locally.

___
NOTE:
You can even run Vowpal Wabbit inside the stacker with the VWClasssifer and VWRegression sklearn wrappers
from `pip install vowpalwabbit`