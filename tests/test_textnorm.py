from voice_agent.textnorm import normalize, prepare


def test_plain_number():
    assert normalize("у нас 21 мастер") == "у нас двадцать один мастер"


def test_grouped_number():
    assert normalize("2 000 рублей") == "две тысячи рублей"


def test_time():
    assert normalize("в 14:00") == "в четырнадцать ноль-ноль"
    assert normalize("в 9:05") == "в девять пять"


def test_date_uses_genitive_ordinal():
    assert normalize("3 сентября") == "третьего сентября"


def test_hyphenated_ordinal():
    assert normalize("25-го числа") == "двадцать пятого числа"


def test_prepare_strips_markdown():
    assert prepare("**Хорошо**, до `встречи`") == "Хорошо , до встречи".replace(" ,", ",")
