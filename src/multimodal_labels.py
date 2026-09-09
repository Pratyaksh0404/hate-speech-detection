"""
Label normalization for the MultiOFF multimodal dataset.

The raw CSVs use string labels "offensive" and "Non-offensiv" (that
second one is missing its final "e" in the source data itself -- a real
typo we found during inspection, not something introduced here). This
module normalizes both to a clean binary 0/1 scheme, tolerant of casing/
whitespace/the known typo.

Convention: offensive=1, non-offensive=0 -- matching the common convention
in the hate-speech/offensive-content literature (including the original
MultiOFF paper) of treating the class of interest (offensive content) as
the positive class.
"""

_LABEL_TO_INT = {
    "offensive": 1,
    "non-offensive": 0,
    "non-offensiv": 0,  # the actual typo present in the raw MultiOFF CSVs
}

INT_TO_LABEL_NAME = {0: "Non-offensive", 1: "Offensive"}


def normalize_label(raw_label) -> int:
    """
    Maps a raw MultiOFF label string to a clean 0/1 int. Raises on any
    value that isn't a recognized variant, rather than silently guessing --
    a new/unexpected label string should be surfaced, not swallowed.
    """
    key = str(raw_label).strip().lower()
    if key not in _LABEL_TO_INT:
        raise ValueError(
            f"Unrecognized MultiOFF label: {raw_label!r}. "
            f"Expected one of: {sorted(_LABEL_TO_INT.keys())}"
        )
    return _LABEL_TO_INT[key]
