from app.utils import normalize_text

def test_accented_and_unicode():
    s = "ação cafés — São Paulo\n\n"  # contains accented chars and newline
    out = normalize_text(s)
    assert 'ação' in out
    assert 'cafés' in out
    assert '\n' not in out

def test_multiple_spaces_and_lines():
    s = "isso   tem    muitos\n\nespacos\t\t e tabs"
    out = normalize_text(s)
    assert '  ' not in out  # no double spaces
    assert '\n' not in out
    assert out.strip() == out

def test_emojis_preserved():
    s = "Amei 😄😄!!!   \n\n Vamos?" 
    out = normalize_text(s)
    assert '😄' in out
    assert '!!!' in out
