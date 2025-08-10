import numpy as np
from sklearn.preprocessing import FunctionTransformer


population_range = [
    "< 100 mil",
    "100 - 250 mil",
    "250 - 500 mil",
    "> 500 mil",
]

age_group = [
    # None,
    "< 1 ano",
    "1 a 4 anos",
    "5 a 11 anos",
    "12 a 15 anos",
    "16 a 18 anos",
    "19 a 24 anos",
    "25 a 29 anos",
    "30 a 39 anos",
    "40 a 49 anos",
    "50 a 59 anos",
    "60 a 65 anos",
    "66 a 85 anos",
    "Maior que 85 anos",
]


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
