# A time saving utility file set to run against _ALL_ sklearn classifers
# If the data set is large it may run into memory issues as all final models are stored in memory.

from sklearn.utils.testing import all_estimators
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GMM, DPGMM, BayesianGaussianMixture, VBGMM
from sklearn.svm import NuSVC, SVC

# Useful for seeing all sklearn estimators that have `predict_prob` attribute
estimators = all_estimators()
for name, class_ in estimators:
    if hasattr(class_, 'predict_proba'):
        print(name)

# Now pick and choose the ones you like
estimators = {AdaBoostClassifier(): 'AdaBoost',
              BayesianGaussianMixture(): 'BayesianGaussianMixture',
              BernoulliNB(): 'BernoulliNB',
              DPGMM(): 'DPGMM',
              ExtraTreesClassifier(): 'ExtraTreesClassifier',
              GMM(): 'GMM',
              GaussianNB(): 'GaussianNB',
              GaussianProcessClassifier(): 'GaussianProcessClassifier',
              GradientBoostingClassifier(): 'GradientBoostingClassifier',
              KNeighborsClassifier(): 'KNeighborsClassifier',
              LabelPropagation(): 'LabelPropagation',
              LabelSpreading(): 'LabelSpreading',
              LinearDiscriminantAnalysis(): 'LinearDiscriminantAnalysis',
              LogisticRegression(): 'LogisticRegression',
              MLPClassifier(): 'MLPClassifier',
              NuSVC(): 'NuSVC',
              QuadraticDiscriminantAnalysis(): 'QuadraticDiscriminantAnalysis',
              RandomForestClassifier(): 'RandomForestClassifier',
              SGDClassifier(): 'SGDClassifier',
              SVC(): 'SVC',
              VBGMM(): 'VBGMM'}

