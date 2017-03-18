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
4. Support `scipy.sparse` data to allow both dense and sparse models to be run on the same folds
5. Create an example of bayesian encoding as a transformer that runs across folds in line with the stacker.
6. Create Hyper-parameter autotuning class for the set of base models to be used in the metaleaner

As with all OSS projects things are under constant development and already one can see places where refactoring some
of the core code would make sense (i.e a folds Class to remove the cumbersome _predict_# and _fit_# methods)



## Install 
```python

pip install -U git+https://github.com/mpearmain/gestalt
```

## Example 
In the examples dir there is a set of notebooks showing various different use cases.
To run the r wrapper for ranger you need to havea  copy of `rpy2` and a copy of teh `ranger` R library locally.