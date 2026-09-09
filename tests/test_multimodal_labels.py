from src.multimodal_labels import normalize_label


def test_offensive_maps_to_1():
    assert normalize_label("offensive") == 1


def test_non_offensive_typo_maps_to_0():
    # the actual typo present in the raw CSVs
    assert normalize_label("Non-offensiv") == 0


def test_case_and_whitespace_insensitive():
    assert normalize_label("  OFFENSIVE  ") == 1
    assert normalize_label("non-OFFENSIV") == 0


def test_unrecognized_label_raises():
    try:
        normalize_label("something else entirely")
        assert False, "expected ValueError for an unrecognized label"
    except ValueError:
        pass
