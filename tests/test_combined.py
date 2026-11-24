from scripts.evaluate_combined import combine


def test_accepts_oov_compound_inside_single_syllable_run():
    dictionary = {"học_sinh", "học", "sinh"}
    bigram = ["học_sinh", "khủng", "bố"]
    lstm = ["học", "sinh", "khủng_bố"]
    assert combine(bigram, lstm, dictionary) == ["học_sinh", "khủng_bố"]


def test_keeps_dictionary_decision_for_known_words():
    dictionary = {"nhà", "hàng_xóm", "nhà_hàng", "xóm"}
    bigram = ["nhà", "hàng_xóm"]
    lstm = ["nhà_hàng", "xóm"]
    assert combine(bigram, lstm, dictionary) == ["nhà", "hàng_xóm"]
