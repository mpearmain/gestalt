# A time saving utility file set to run against _ALL_ sklearn classifers
# If the daa set is large it may run into memory issues as all final models are stored in memory.

estimators = {RandomForestClassifier(): 'RFC',
              ExtraTreesClassifier(): 'ETC',
              XGBClassifier(): 'XGB1'}
