import pytest

from voice_agent import llm


class Dead:
    name = "dead"

    def stream(self, messages):
        raise RuntimeError("down")
        yield


class Half:
    name = "half"

    def stream(self, messages):
        yield "Здрав"
        raise RuntimeError("broke mid-sentence")


class Fixed:
    name = "fixed"

    def stream(self, messages):
        yield "ok"


def test_echo():
    assert "".join(llm.make("echo").stream([{"role": "user", "content": "привет"}])) == "Вы сказали: привет"


def test_chain_falls_back_before_first_word():
    chain = llm.Chain([Dead(), Fixed()])
    assert "".join(chain.stream([])) == "ok"
    assert chain.last_used == "fixed"


def test_chain_does_not_restart_after_first_word():
    chain = llm.Chain([Half(), Fixed()])
    got = []
    with pytest.raises(RuntimeError):
        for piece in chain.stream([]):
            got.append(piece)
    assert got == ["Здрав"]          # the caller never hears a second, different answer


def test_unknown_provider_is_an_error():
    with pytest.raises(RuntimeError):
        llm.make("nope")
