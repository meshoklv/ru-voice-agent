"""Russian text normalisation for TTS: numbers, dates and times to words.

Most TTS engines read "15 750" as "15" and "750", "14:00" as garbage and "3 сентября"
with a cardinal instead of the genitive ordinal. This is a small regex layer on top
of ``num2words``. It is a pragmatic helper, not a full grammar engine: cases beyond
dates, times, hyphenated ordinals and plain integers are left alone.
"""
import re

try:
    from num2words import num2words
except ImportError:                                   # pragma: no cover
    num2words = None

MONTHS_GEN = ("января февраля марта апреля мая июня июля августа "
              "сентября октября ноября декабря").split()

_IRREGULAR_THIRD = {"го": "третьего", "му": "третьему", "м": "третьем",
                    "х": "третьих", "я": "третья", "е": "третье", "й": "третий"}


def ordinal_with_ending(n: int, ending: str) -> str:
    """Nominative ordinal from num2words with the last word re-ended for the case."""
    base = num2words(n, lang="ru", to="ordinal")
    words = base.split(" ")
    last = words[-1]
    if last == "третий":
        words[-1] = _IRREGULAR_THIRD.get(ending, last)
        return " ".join(words)
    stem = last[:-2]
    for suf in ("ий", "ый", "ой"):
        if last.endswith(suf):
            stem = last[:-len(suf)]
            break
    mapping = {"го": stem + "ого", "му": stem + "ому", "м": stem + "ом",
               "х": stem + "ых", "я": stem + "ая", "е": stem + "ое", "й": last}
    words[-1] = mapping.get(ending, last)
    return " ".join(words)


def normalize(text: str) -> str:
    if num2words is None:
        return text
    out = text
    # "3 сентября" -> "третьего сентября"
    out = re.sub(r"\b(\d{1,2}) (%s)\b" % "|".join(MONTHS_GEN),
                 lambda m: "%s %s" % (ordinal_with_ending(int(m.group(1)), "го"),
                                      m.group(2)), out)
    # "25-го", "3-я"
    out = re.sub(r"\b(\d{1,3})-(го|му|м|х|я|е|й)\b",
                 lambda m: ordinal_with_ending(int(m.group(1)), m.group(2)), out)

    # "14:00" -> "четырнадцать ноль-ноль"
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        hw = num2words(h, lang="ru")
        return "%s ноль-ноль" % hw if mi == 0 else "%s %s" % (hw, num2words(mi, lang="ru"))
    out = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time, out)
    # "15 750" -> "15750"
    out = re.sub(r"(?<=\d) (\d{3})(?!\d)", r"\1", out)
    # remaining integers -> words
    return re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="ru"), out)


def prepare(text: str) -> str:
    """Strip markdown-ish symbols and normalise numbers: text ready to be spoken."""
    text = re.sub(r"[*_`#>\[\]]+", "", text)
    text = re.sub(r"\s+([,.!?…])", r"\1", re.sub(r"\s+", " ", text)).strip()
    try:
        return normalize(text)
    except Exception:                                  # noqa: BLE001
        return text
