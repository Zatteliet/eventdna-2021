from pathlib import Path
from experiments.featurizer import get_sentences, featurize_lets_sentence


def test_read_lets():
    case = Path("tests/data/lets.csv")
    assert case.exists()
    sentences = get_sentences(case)
    assert next(sentences) == [
        ["Provincie", "provincie", "N(soort,ev,basis,zijd,stan)", "B-NP", "O"],
        [
            "Antwerpen",
            "Antwerpen",
            "N(eigen,ev,basis,onz,stan)",
            "I-NP",
            "B-LOC",
        ],
        ["telt", "tellen", "WW(pv,tgw,met-t)", "B-VP", "O"],
        ["elke", "elk", "VNW(onbep,det)", "B-NP", "O"],
        ["drie", "drie", "TW(hoofd,prenom,stan)", "I-NP", "O"],
        ["weken", "week", "N(soort,mv,basis)", "I-NP", "O"],
        ["een", "een", "LID(onbep)", "B-NP", "O"],
        ["dodelijk", "dodelijk", "ADJ(prenom,basis,zonder)", "I-NP", "O"],
        [
            "arbeidsongeval",
            "arbeidsongeval",
            "N(soort,ev,basis,onz,stan)",
            "I-NP",
            "O",
        ],
    ]


def test_featurize_lets_sentence():
    case = [
        ["Provincie", "provincie", "N(soort,ev,basis,zijd,stan)", "B-NP", "O"],
        [
            "Antwerpen",
            "Antwerpen",
            "N(eigen,ev,basis,onz,stan)",
            "I-NP",
            "B-LOC",
        ],
        ["telt", "tellen", "WW(pv,tgw,met-t)", "B-VP", "O"],
    ]
    feats = list(featurize_lets_sentence(case))
    for token_feat_dict in feats:
        assert "token" in token_feat_dict.keys()
        print(token_feat_dict)
