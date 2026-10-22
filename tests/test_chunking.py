from voice_agent.chunking import pop_speakable


def test_waits_for_sentence_end():
    assert pop_speakable("Здравствуйте, это") == ("", "Здравствуйте, это")


def test_cuts_complete_sentence():
    ready, rest = pop_speakable("Хорошо. Записал вас")
    assert ready == "Хорошо." and rest == " Записал вас"


def test_force_flushes_tail():
    assert pop_speakable("Записал вас", force=True) == ("Записал вас", "")
    assert pop_speakable("   ", force=True) == ("", "   ")


def test_first_chunk_goes_out_at_a_comma():
    ready, rest = pop_speakable("Здравствуйте, рады вас слышать и готовы помочь", first=True)
    assert ready == "Здравствуйте, рады вас слышать и готовы помочь"[:len(ready)]
    assert ready.endswith(",") is False or len(ready) >= 18
    assert not pop_speakable("Да, ", first=True)[0]        # shorter than FIRST_CHUNK_MIN


def test_decimal_point_is_not_a_sentence_end():
    assert pop_speakable("Это стоит 2.5 тысячи")[0] == ""
