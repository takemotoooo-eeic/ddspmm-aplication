export const SAMPLE_RATE = 16000;
export const BYTES_PER_SAMPLE = 2;
export const TIME_SCALE = 200; // px per second (fixed, no zoom)

/** TTM (MelodyFlow) で編集できる区間の最大長 [s] */
export const TTM_MAX_SEGMENT_SEC = 30;

/** 波形ビューポート端付近でドラッグ中に横スクロールを開始する余白 [px] */
export const TTM_EDGE_SCROLL_MARGIN_PX = 56;

/** 端スクロール時の最大速度（1 フレームあたりの scrollLeft 変化量）[px] */
export const TTM_EDGE_SCROLL_SPEED_PX = 22;

export const LOUDNESS_MIN_DB = -100;
export const LOUDNESS_MAX_DB = 0;
export const LOUDNESS_EDITOR_HEIGHT = 480;

export const NOTE_HEIGHT = 30;
export const PIANO_ROLL_KEY_WIDTH = 80;
export const TIMELINE_HEIGHT = 20;

export const PITCH_SAMPLE_RATE = 31.25;

export const INSTRUMENTS = ['va', 'vn', 'vc', 'fl', 'cl', 'ob'] as const;
export type Instrument = (typeof INSTRUMENTS)[number];
