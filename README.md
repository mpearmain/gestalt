# Gestalt

A helper library for data science pipeline

"_Something that is made of many parts and yet is somehow more than or different from the combination of its parts_"

Machine learning (ML) has achieved considerable success in multiple fields, much of the pipeline surrounding machine 
learning can be generalised to a point that removes the 'engineering' of the data flows and pipelines leaving the human
expert to work on the most difficult aspects:

1. Pre-processing the data
2. Feature Engineering
3. Select appropriate features

## The goals of Gestalt
The under pining technology of Gestalt is the fantastic python package DASK (http://dask.pydata.org) 

"`Dask is a flexible parallel computing library for analytic computing.`"

### Programming by Optimization
Following the paradigm of Programming by Optimization, we aim for a plug-able design to allow different algorithms (and
indeed different software) to be instantiated automatically to provide teh best solution to the problem space.