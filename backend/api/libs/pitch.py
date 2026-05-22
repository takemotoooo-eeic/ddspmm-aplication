import numpy as np

def hz_to_cent(hz: np.ndarray) -> np.ndarray:
    hz = np.maximum(hz, 1e-7)
    return 1200 * np.log2(hz / 440.0)

def cent_to_hz(cent: np.ndarray) -> np.ndarray:
    return 440.0 * (2 ** (cent / 1200.0))

def hz_to_midi(hz: float) -> float:
    return 12 * np.log2(hz / 440.0) + 69

def midi_to_hz(midi: int) -> float:
    return 440.0 * (2 ** ((float(midi) - 69.0) / 12.0))


def midi_to_cent(midi: int) -> float:
    return 1200 * np.log2(midi_to_hz(midi) / 440.0)
