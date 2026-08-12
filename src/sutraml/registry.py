# Linear models
from sklearn.cross_decomposition import PLSRegression
# Ensemble
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
# Other regressors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor,
                                  Lars, Lasso, LassoLars, LinearRegression,
                                  LogisticRegression,
                                  OrthogonalMatchingPursuit,
                                  PassiveAggressiveRegressor, RANSACRegressor,
                                  Ridge, RidgeClassifier, SGDClassifier,
                                  SGDRegressor, TheilSenRegressor)
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# KNN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
# SVM
from sklearn.svm import SVC, SVR
# Tree-based
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# Boosting libraries
from xgboost import (XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor)
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# ------------------------
# MODEL REGISTRY
# ------------------------
MODEL_REGISTRY = {
    "classification": {
        "LR"    : ("Logistic Regression", LogisticRegression),
        "RDG"   : ("Ridge Classifier", RidgeClassifier),
        "SGD"   : ("SGD Classifier", SGDClassifier),
        "NB"    : ("Naive Bayes", GaussianNB),
        "SVM"   : ("Support Vector Machine", SVC),
        "KNN"   : ("K-Neighbors Classifier", KNeighborsClassifier),
        "DT"    : ("Decision Tree", DecisionTreeClassifier),
        "RF"    : ("Random Forest", RandomForestClassifier),
        "ETC"   : ("Extra Trees Classifier", ExtraTreesClassifier),
        "GB"    : ("Gradient Boosting", GradientBoostingClassifier),
        "AB"    : ("AdaBoost", AdaBoostClassifier),
        "HGB"   : ("HistGradient Boosting", HistGradientBoostingClassifier),
        "XGB"   : ("XGBoost", XGBClassifier),
        "XGBRF" : ("XGB (Random Forest)", XGBRFClassifier),
        "LGBC"  : ("LGBM Classifier", LGBMClassifier),
        "CB"    : ("CatBoost", CatBoostClassifier),
        # Lasso/ElasticNet can be used with LogisticRegression wrapper for feature selection
        "LC"    : ("Lasso", Lasso),
        "ENC"   : ("ElasticNet", ElasticNet),
    },
    "regression": {
        "LR"     : ("Linear Regression", LinearRegression),
        "RR"     : ("Ridge Regression", Ridge),
        "SGR"    : ("SGD Regressor", SGDRegressor),
        "BRR"    : ("BayesianRidge", BayesianRidge),
        "HR"     : ("Huber Regressor", HuberRegressor),
        "LAR"    : ("Lars", Lars),
        "LLR"    : ("Lasso Lars", LassoLars),
        "OMPR"   : ("Orthogonal Matching Pursuit", OrthogonalMatchingPursuit),
        "PAR"    : ("Passive Aggressive", PassiveAggressiveRegressor),
        "TSR"    : ("Theil Sen", TheilSenRegressor),
        "RANR"   : ("RANSAC", RANSACRegressor),
        "SVR"    : ("Support Vector Machine", SVR),
        "KNN"    : ("K-Neighbors Regressor", KNeighborsRegressor),
        "DTR"    : ("Decision Tree", DecisionTreeRegressor),
        "RFR"    : ("Random Forest", RandomForestRegressor),
        "ETR"    : ("Extra Trees", ExtraTreesRegressor),
        "GBR"    : ("Gradient Boosting", GradientBoostingRegressor),
        "ABR"    : ("Ada Boost", AdaBoostRegressor),
        "HGR"    : ("HistGradient Boosting", HistGradientBoostingRegressor),
        "GPR"    : ("Gaussian Process", GaussianProcessRegressor),
        "IR"     : ("Isotonic", IsotonicRegression),
        "KRR"    : ("Kernel Ridge", KernelRidge),
        "PLSR"   : ("PLS Regression", PLSRegression),
        "MLPR"   : ("MLP Regressor", MLPRegressor),
        "XGBR"   : ("XG Boost", XGBRegressor),
        "XGBRFR" : ("XGB (Random Forest)", XGBRFRegressor),
        "LGBR"   : ("LightGBM", LGBMRegressor),
        "CBR"    : ("CatBoost", CatBoostRegressor),
        # Lasso/ElasticNet for regression as well
        "LSR"    : ("Lasso Regresion", Lasso),
        "ENR"    : ("ElasticNet", ElasticNet),
    }
}
