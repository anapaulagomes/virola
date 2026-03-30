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
        "region": "Centro-Oeste",
        "population_range": "> 500 mil",
    },
    520110: {"city": "Anápolis", "region": "Centro-Oeste", "population_range": "250 - 500 mil"},
    431490: {"city": "Porto Alegre", "region": "Sul", "population_range": "> 500 mil"},
    430920: {"city": "Gravataí", "region": "Sul", "population_range": "250 - 500 mil"},
    354980: {"city": "Bauru", "region": "Sudeste", "population_range": "250 - 500 mil"},
    310620: {"city": "Belo Horizonte", "region": "Sudeste", "population_range": "> 500 mil"},
    292740: {"city": "Salvador", "region": "Nordeste", "population_range": "> 500 mil"},
    230370: {"city": "Caucaia", "region": "Nordeste", "population_range": "250 - 500 mil"},
    130260: {"city": "Manaus", "region": "Norte", "population_range": "> 500 mil"},
    120040: {"city": "Rio Branco", "region": "Norte", "population_range": "250 - 500 mil"},
}

CITY_NAMES = [city["city"] for city in IBGE_CITY_MAP.values()]
