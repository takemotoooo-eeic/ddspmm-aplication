export const SAMPLE_RATE = 16000;
export const BYTES_PER_SAMPLE = 2;
export const TIME_SCALE = 200; // px per second (fixed, no zoom)

export const LOUDNESS_MIN_DB = -80;
export const LOUDNESS_MAX_DB = -20;
export const LOUDNESS_EDITOR_HEIGHT = 480;

export const NOTE_HEIGHT = 30;
export const PIANO_ROLL_KEY_WIDTH = 80;
export const TIMELINE_HEIGHT = 20;

export const PITCH_SAMPLE_RATE = 31.25;

export const INSTRUMENTS = ['va', 'vn', 'vc', 'fl', 'cl', 'ob'] as const;
export type Instrument = (typeof INSTRUMENTS)[number];
