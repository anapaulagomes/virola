POPULATION_RANGE = [
    "< 100 mil",
    "100 - 250 mil",
    "250 - 500 mil",
    "> 500 mil",
]

ORIGINAL_AGE_GROUP = [
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

AGE_GROUP = [
    "< 1 ano",
    "1 a 15 anos",
    "16 a 59 anos",
    "Maior que 60 anos",
]

IBGE_CITY_MAP = {
    530010: {
        "city": "Distrito Federal",
        "state": "DF",
        "region": "Centro-Oeste",
        "population_range": "> 500 mil",
        "lat": -15.793889,
        "lon": -47.882778,
    },
    520110: {
        "city": "Anápolis",
        "state": "GO",
        "region": "Centro-Oeste",
        "population_range": "250 - 500 mil",
        "lat": -16.326667,
        "lon": -48.952778,
    },
    431490: {
        "city": "Porto Alegre",
        "state": "RS",
        "region": "Sul",
        "population_range": "> 500 mil",
        "lat": -30.033056,
        "lon": -51.230000,
    },
    430920: {
        "city": "Gravataí",
        "state": "RS",
        "region": "Sul",
        "population_range": "250 - 500 mil",
        "lat": -29.934722,
        "lon": -50.993889,
    },
    354980: {
        "city": "Bauru",
        "state": "SP",
        "region": "Sudeste",
        "population_range": "250 - 500 mil",
        "lat": -22.314722,
        "lon": -49.060556,
    },
    310620: {
        "city": "Belo Horizonte",
        "state": "MG",
        "region": "Sudeste",
        "population_range": "> 500 mil",
        "lat": -19.916667,
        "lon": -43.933333,
    },
    292740: {
        "city": "Salvador",
        "state": "BA",
        "region": "Nordeste",
        "population_range": "> 500 mil",
        "lat": -12.974722,
        "lon": -38.476667,
    },
    230370: {
        "city": "Caucaia",
        "state": "CE",
        "region": "Nordeste",
        "population_range": "250 - 500 mil",
        "lat": -3.731944,
        "lon": -38.654167,
    },
    130260: {
        "city": "Manaus",
        "state": "AM",
        "region": "Norte",
        "population_range": "> 500 mil",
        "lat": -3.118889,
        "lon": -60.021667,
    },
    120040: {
        "city": "Rio Branco",
        "state": "AC",
        "region": "Norte",
        "population_range": "250 - 500 mil",
        "lat": -9.974722,
        "lon": -67.810000,
    },
}

CITY_NAMES = [city["city"] for city in IBGE_CITY_MAP.values()]

ORIGINAL_TO_REDUCED_AGE_GROUP = {
    "< 1 ano": "< 1 ano",
    "1 a 4 anos": "1 a 15 anos",
    "5 a 11 anos": "1 a 15 anos",
    "12 a 15 anos": "1 a 15 anos",
    "16 a 18 anos": "16 a 59 anos",
    "19 a 24 anos": "16 a 59 anos",
    "25 a 29 anos": "16 a 59 anos",
    "30 a 39 anos": "16 a 59 anos",
    "40 a 49 anos": "16 a 59 anos",
    "50 a 59 anos": "16 a 59 anos",
    "60 a 65 anos": "Maior que 60 anos",
    "66 a 85 anos": "Maior que 60 anos",
    "Maior que 85 anos": "Maior que 60 anos",
}


CODE_TYPES_MAPPING = {
    "CID": "cid10",
    "cid": "cid10",
    "CID 10": "cid10",
    "CIAP": "ciap2",
    "ciap": "ciap2",
    "AB": "ab",
    "ABP": "ab",
}
