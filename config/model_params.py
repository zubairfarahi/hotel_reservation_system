from scipy.stats import randint, uniform

LIGHTGM_PARAMS = {
    'max _depth' : randint (5,50),
    '"Lestimators' :randint (100,500),
    'learning_rate': uniform (0.01,0.2),
    'num_leaves': randint(20,100),
    'boosting_type' : ['gbdt','dart','goss']
}

RANDOM_SEARCH_PARAMS = {
    'n_iter' : 4,
    'cv' : 2,
    'n_jobs': -1,
    'verbose' :2,
    'random_state' : 42,
    'scoring' : 'accuracy'
}