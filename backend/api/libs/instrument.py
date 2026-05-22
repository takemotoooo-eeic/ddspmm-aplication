import os
from enum import Enum


class Instrument(str, Enum):
    CL = "cl"
    FL = "fl"
    OB = "ob"
    SAX = "sax"
    TPT = "tpt"
    VN = "vn"
    TBN = "tbn"
    VC = "vc"
    HN = "hn"
    TBA = "tba"
    VA = "va"
    BN = "bn"
    DB = "db"
    TIMP = "timp"


_URMP_INSTRUMENT_CODES = {instrument.value for instrument in Instrument}


def parse_instrument_names_from_urmp_filename(filename: str | None) -> list[str] | None:
    """URMP形式のパス/ファイル名から楽器略称を抽出する。

    例:
        AuMix_19_Pavane_cl_vn.wav -> ["cl", "vn"]
        19_Pavane_cl_vn/AuMix_19_Pavane_cl_vn.wav -> ["cl", "vn"]
    """
    if not filename:
        return None

    candidates = [
        os.path.splitext(os.path.basename(filename))[0],
        os.path.basename(os.path.dirname(filename)),
    ]

    for name in candidates:
        if not name or name in (".", ".."):
            continue
        parts = name.split("_")
        instruments: list[str] = []
        for part in reversed(parts):
            code = part.lower()
            if code not in _URMP_INSTRUMENT_CODES:
                break
            instruments.append(code)
        if instruments:
            instruments.reverse()
            return instruments
    return None
