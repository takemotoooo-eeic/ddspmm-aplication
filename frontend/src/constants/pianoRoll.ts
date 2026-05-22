export const keys = [
  { note: 'B', isBlack: false },
  { note: 'A#', isBlack: true },
  { note: 'A', isBlack: false },
  { note: 'G#', isBlack: true },
  { note: 'G', isBlack: false },
  { note: 'F#', isBlack: true },
  { note: 'F', isBlack: false },
  { note: 'E', isBlack: false },
  { note: 'D#', isBlack: true },
  { note: 'D', isBlack: false },
  { note: 'C#', isBlack: true },
  { note: 'C', isBlack: false },
] as const;

export const octaves = [6, 5, 4, 3, 2, 1] as const;

export const TOTAL_KEYS = octaves.length * keys.length;
export const PIANO_ROLL_HEIGHT = TOTAL_KEYS * 30;
