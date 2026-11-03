from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Rough_data"

TWEET_ROWS_TO_KEEP = ['lang', 'created_at', 'text', 'public_metrics_retweet_count', 
                      'public_metrics_reply_count', 'public_metrics_like_count', 'public_metrics_quote_count', 
                      'public_metrics_bookmark_count', 'geo_place_id', 'geo_coordinates.coordinates',
                      'hashtags', 'context_domains',
                      ]

# Geneva Coordinate
LAT_G, LON_G = 46.2044, 6.1432


SHORT_LIST = ['1008', '1502']


###########################################################
## Entities to track in the fuzzy matching approach
###########################################################

ENTITY_LEXICON = {
    # ============================================================
    # SYMPTOMS
    # ============================================================
    "fever": {
        "fr": [
            "fièvre", "fievre", "fiévreux", "fiévreuse",
            "fébrile",
            "température", "tempé",
            "j'ai de la fièvre", "fait de la fièvre",
            "monte en température", "monter en température",
            "38°", "38 °c", "38 degrés",
            "39°", "39 °c", "39 degrés",
            "40°", "40 °c", "40 degrés",
        ],
        "de": [
            "fieber", "fiebert", "fieberhaft", "fieberanfall",
            "fieberschub", "erhöhte temperatur",
            "habe fieber", "hat fieber", "mit fieber",
            "38 grad", "39 grad", "40 grad",
        ],
        "en": [
            "fever", "feverish", "fevers",
            "high temperature", "high temp",
            "running a temp", "running a fever",
            "got a fever", "have a fever", "had a fever",
            "high fever", "low-grade fever", "low grade fever",
            "38 degrees", "39 degrees", "40 degrees",
        ],
    },

    "cough": {
        "fr": [
            "toux", "tousse", "tousser", "toussais", "toussait",
            "toussé", "toussée", "toussotement",
            "je tousse", "il tousse", "elle tousse",
            "quinte de toux", "quintes de toux",
            "toux sèche", "toux grasse",
        ],
        "de": [
            "husten", "hustet", "hustete", "gehustet",
            "hustenanfall", "reizhusten", "trockener husten",
            "habe husten", "hat husten", "muss husten",
        ],
        "en": [
            "cough", "coughing", "coughed", "coughs",
            "dry cough", "persistent cough", "bad cough",
            "can't stop coughing", "keep coughing",
        ],
    },

    "sore_throat": {
        "fr": [
            "mal de gorge", "maux de gorge", "mal au cou",
            "j'ai mal à la gorge", "gorge qui gratte",
            "gorge irritée", "gorge enflammée",
            "angine", "pharyngite",
        ],
        "de": [
            "halsschmerzen", "halsweh",
            "kratzen im hals", "rachen tut weh",
            "wunder hals", "entzündeter hals",
        ],
        "en": [
            "sore throat", "scratchy throat", "throat hurts",
            "throat is killing me", "painful throat",
            "throat pain",
        ],
    },

    "headache": {
        "fr": [
            "mal de tête", "maux de tête",
            "j'ai mal à la tête",
            "migraine", "céphalée", "céphalées",
        ],
        "de": [
            "kopfschmerzen", "kopfweh", "migräne",
            "kopf tut weh", "habe kopfschmerzen",
        ],
        "en": [
            "headache", "headaches", "migraine",
            "head is killing me", "head pain",
            "splitting headache", "bad headache",
        ],
    },

    "fatigue": {
        "fr": [
            # Strong illness markers
            "fatigue intense", "grosse fatigue", "fatigue extrême",
            "épuisement", "épuisé", "épuisée",
            "exténué", "exténuée",
            "asthénie",
            # Common but noisier
            "fatigué", "fatiguée", "fatigue",
            "à plat", "complètement vidé", "complètement vidée",
            "lessivé", "lessivée",
            "fatigue chronique",
        ],
        "de": [
            # Strong illness markers
            "erschöpfung", "erschöpft",
            "starke müdigkeit", "extreme müdigkeit",
            "abgeschlagen", "abgeschlagenheit",
            "kraftlos", "ausgelaugt",
            # Common but noisier
            "müde", "schlapp",
            "fertig", "kaputt",
        ],
        "en": [
            # Strong illness markers
            "exhaustion", "exhausted",
            "extreme fatigue", "severe fatigue", "intense fatigue",
            "completely drained", "wiped out",
            # Common but noisier
            "fatigue", "fatigued",
            "drained", "no energy", "low energy",
            "dead tired", "knackered",
        ],
    },

    "breathing_difficulty": {
        "fr": [
            # Colloquial — bulk of useful signal
            "essoufflé", "essoufflée", "essoufflement",
            "à bout de souffle", "souffle court",
            "j'arrive plus à respirer", "j'arrive pas à respirer",
            "difficile de respirer", "difficulté à respirer",
            "difficultés respiratoires",
            "manque d'air", "manque de souffle",
            "respiration difficile",
            # Clinical
            "dyspnée", "dyspneique",
        ],
        "de": [
            # Colloquial
            "atemnot", "kurzatmig", "kurzatmigkeit",
            "ausser atem", "außer atem",
            "schwer atmen", "schwer zu atmen",
            "atembeschwerden", "atemprobleme",
            "bekomme keine luft", "keine luft",
            # Clinical
            "dyspnoe", "dyspnoeisch",
        ],
        "en": [
            # Colloquial — most signal
            "shortness of breath", "short of breath",
            "out of breath", "can't breathe", "cannot breathe",
            "trouble breathing", "difficulty breathing",
            "hard to breathe", "struggling to breathe",
            "breathing problems", "breathing issues",
            "winded",
            # Clinical
            "dyspnea", "dyspnoea", "dyspneic",
        ],
    },

    "anosmia": {
        "fr": [
            "anosmie", "perte d'odorat", "perte de l'odorat",
            "perte de l'olfaction",
            "plus d'odorat", "je n'ai plus d'odorat",
            "je ne sens plus rien", "ne sens plus rien",
            "je sens plus rien",
            "ne sens plus les odeurs",
        ],
        "de": [
            "anosmie", "geruchsverlust", "kein geruchssinn",
            "verlust des geruchssinns",
            "rieche nichts mehr", "rieche nichts",
            "kann nichts riechen", "kann nicht mehr riechen",
        ],
        "en": [
            "anosmia", "loss of smell", "lost my smell",
            "lost sense of smell", "loss of sense of smell",
            "can't smell", "cannot smell", "no sense of smell",
            "smell is gone", "lost smell",
        ],
    },

    "ageusia": {
        "fr": [
            "agueusie", "perte du goût", "perte de gout",
            "perte de goût", "perte du gout",
            "plus de goût", "plus de gout",
            "ne sens plus le goût", "ne sens plus le gout",
            "ne sens plus rien au niveau du goût",
        ],
        "de": [
            "ageusie", "geschmacksverlust", "kein geschmack",
            "verlust des geschmackssinns",
            "schmecke nichts", "schmecke nichts mehr",
            "kann nichts schmecken",
        ],
        "en": [
            "ageusia", "loss of taste", "lost my taste",
            "lost sense of taste", "loss of sense of taste",
            "can't taste", "cannot taste", "no sense of taste",
            "taste is gone",
        ],
    },

    # ============================================================
    # TESTING
    # ============================================================
    "positive_test": {
        "fr": [
            # Direct positive test phrasings
            "test positif", "testé positif", "testée positive",
            "résultat positif",
            "je suis positif", "je suis positive",
            "je suis testé positif", "je suis testée positive",
            "positif au covid", "positive au covid",
            "positif au coronavirus",
            # "Got COVID" idioms
            "j'ai le covid", "j'ai eu le covid", "j'ai la covid",
            "j'ai chopé le covid", "j'ai attrapé le covid",
            "j'ai chopé la covid", "j'ai attrapé la covid",
            "j'ai chopé le corona", "j'ai attrapé le corona",
            # Confirmed cases
            "covid confirmé", "cas confirmé",
        ],
        "de": [
            "positiv getestet", "positives ergebnis",
            "bin positiv", "test positiv",
            "positiv auf corona", "positiv auf covid",
            # "Got COVID" idioms
            "habe corona", "habe covid", "habe das virus",
            "corona erwischt", "covid erwischt",
            "hatte corona", "hatte covid",
            # Confirmed
            "corona bestätigt", "covid bestätigt",
        ],
        "en": [
            "tested positive", "positive test", "positive result",
            "i'm positive", "i am positive",
            "positive for covid", "positive for corona",
            "got covid", "have covid", "had covid",
            "got corona", "have corona", "caught covid",
            "caught corona", "covid confirmed",
        ],
    },

    "negative_test": {
        "fr": [
            "test négatif", "testé négatif", "testée négative",
            "résultat négatif",
            "je suis négatif", "je suis négative",
            "négatif au covid", "négative au covid",
        ],
        "de": [
            "negativ getestet", "negatives ergebnis",
            "bin negativ", "test negativ",
            "negativ auf corona", "negativ auf covid",
        ],
        "en": [
            "tested negative", "negative test", "negative result",
            "i'm negative", "i am negative",
            "negative for covid", "negative for corona",
        ],
    },

    # ============================================================
    # BEHAVIORAL
    # ============================================================
    "quarantine": {
        "fr": [
            "quarantaine", "en quarantaine",
            "isolement", "auto-isolement",
            "confiné", "confinée", "confinement",
            "je m'isole", "isolement à domicile",
        ],
        "de": [
            "quarantäne", "in quarantäne",
            "isolation", "selbstisolation",
            "häusliche quarantäne", "in isolation",
        ],
        "en": [
            "quarantine", "in quarantine", "quarantining",
            "self-isolation", "self isolating",
            "isolating", "in isolation", "self-quarantine",
        ],
    },

    "mask_usage": {
        "fr": [
            "masque", "port du masque",
            "masque ffp2", "masque chirurgical", "masque tissu",
            "porter un masque", "porte un masque",
        ],
        "de": [
            "maske", "maskenpflicht", "ffp2-maske", "ffp2 maske",
            "schutzmaske", "mundschutz", "stoffmaske",
            "maske tragen", "trage maske",
        ],
        "en": [
            "mask", "face mask", "wearing a mask", "wear a mask",
            "ffp2", "n95", "surgical mask", "cloth mask",
            "masking", "masked up",
        ],
    },

    # ============================================================
    # VACCINATION
    # ============================================================
    "vaccine": {
        "fr": [
            # General vaccination
            "vaccin", "vacciné", "vaccinée", "vaccination",
            "se vacciner", "me vacciner", "je me vaccine",
            "je me suis vacciné", "je me suis vaccinée",
            "première dose", "1ère dose", "1re dose",
            "deuxième dose", "2ème dose", "2e dose",
            "dose de vaccin",
            # Specific vaccines
            "pfizer", "moderna", "astrazeneca", "janssen",
            "johnson", "comirnaty", "spikevax",
            # Vaccine hesitancy / refusal (epidemiologically relevant)
            "antivax", "anti-vaccin", "pas vacciné", "pas vaccinée",
            "refuse le vaccin", "refus du vaccin",
        ],
        "de": [
            # General vaccination
            "impfung", "geimpft", "impfen", "impfstoff",
            "ich bin geimpft", "wurde geimpft",
            "erste impfung", "erste dosis",
            "zweite impfung", "zweite dosis",
            "covid-impfung", "corona-impfung",
            # Specific vaccines
            "biontech", "pfizer", "moderna", "astrazeneca",
            "johnson", "comirnaty", "spikevax",
            # Vaccine hesitancy / refusal
            "impfgegner", "nicht geimpft", "impfpflicht",
            "impfskepsis",
        ],
        "en": [
            # General vaccination
            "vaccine", "vaccinated", "vaccination", "getting vaccinated",
            "got vaccinated", "got my vaccine", "got my shot",
            "got my jab", "first dose", "second dose",
            "covid vaccine", "covid shot", "covid jab",
            # Specific vaccines
            "pfizer", "moderna", "astrazeneca", "johnson",
            "comirnaty", "spikevax",
            # Vaccine hesitancy / refusal
            "unvaccinated", "not vaccinated", "anti-vax",
            "antivax", "vaccine hesitancy", "refuse the vaccine",
        ],
    },

    "booster": {
        "fr": [
            "rappel", "dose de rappel",
            "troisième dose", "3ème dose", "3e dose",
            "quatrième dose", "4ème dose",
            "booster", "rappel vaccinal",
        ],
        "de": [
            "auffrischung", "auffrischimpfung", "auffrischungsimpfung",
            "boosterimpfung", "booster",
            "dritte dosis", "dritte impfung",
            "vierte dosis", "vierte impfung",
        ],
        "en": [
            "booster", "booster shot", "booster jab",
            "third dose", "third shot", "3rd dose",
            "fourth dose", "fourth shot", "4th dose",
        ],
    },
}


# Lags shared by the tweet features and the dispatch-centre features, so both
# temporal contexts cover the same days (see add_time_series_features and
# add_dispatch_center_features).
FEATURE_LAGS = (1, 3, 7)

NLP_FEATURE_LAGS = (1, 2, 3, 4, 5, 6)
