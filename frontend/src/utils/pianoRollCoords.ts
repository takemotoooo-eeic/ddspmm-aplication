import { NOTE_HEIGHT, PITCH_SAMPLE_RATE } from '../constants/editor';
import {
  keys,
  MIDI_MAX,
  MIDI_MIN,
  octaves,
  PIANO_ROLL_HEIGHT,
  TOTAL_KEYS,
} from '../constants/pianoRoll';
import { hzToMidi, midiToHz, snapMidi } from './pitch';

/** 行上端の Y（SVG rect 用） */
export const midiToRectY = (midi: number): number =>
  midiToY(midi) - NOTE_HEIGHT / 2;

/** ノート矩形用（Hz → 鍵盤範囲内にクランプ） */
export const noteFrequencyToRectY = (frequency: number): number | null => {
  if (!Number.isFinite(frequency) || frequency <= 0) return null;
  const midi = Math.max(MIDI_MIN, Math.min(MIDI_MAX, snapMidi(hzToMidi(frequency))));
  return midiToRectY(midi);
};

/** ノート行の中心 Y */
export const midiToY = (midi: number): number => {
  const rowFromTop = TOTAL_KEYS - 1 - (snapMidi(midi) - MIDI_MIN);
  return rowFromTop * NOTE_HEIGHT + NOTE_HEIGHT / 2;
};

export const yToMidi = (y: number): number => {
  const rowFromTop = (y - NOTE_HEIGHT / 2) / NOTE_HEIGHT;
  const midi = MIDI_MIN + (TOTAL_KEYS - 1) - rowFromTop;
  return snapMidi(midi);
};

export const hzToY = (hz: number): number => {
  const midi = 12 * Math.log2(hz / 440) + 69;
  return midiToY(midi);
};

/** ノート編集用（半音スナップ） */
export const yToHz = (y: number): number => midiToHz(yToMidi(y));

/** ピッチ曲線の手描き用（連続） */
export const yToHzContinuous = (y: number): number => {
  const rowFromTop = (y - NOTE_HEIGHT / 2) / NOTE_HEIGHT;
  const midi = MIDI_MIN + (TOTAL_KEYS - 1) - rowFromTop;
  return midiToHz(midi);
};

export const hzToYContinuous = (hz: number): number => {
  const midi = 12 * Math.log2(hz / 440) + 69;
  const rowFromTop = TOTAL_KEYS - 1 - (midi - MIDI_MIN);
  return rowFromTop * NOTE_HEIGHT + NOTE_HEIGHT / 2;
};

/** 鍵盤表示範囲（C1–B6）内にクランプした描画用 Y */
export const pitchHzToDisplayY = (hz: number): number => {
  const y = hzToYContinuous(hz);
  return Math.max(NOTE_HEIGHT / 2, Math.min(PIANO_ROLL_HEIGHT - NOTE_HEIGHT / 2, y));
};

export const buildPitchPolylinePoints = (
  pitch: number[],
  timeScale: number,
  sampleRate: number = PITCH_SAMPLE_RATE,
): string =>
  pitch
    .map((hz, i) => {
      if (!Number.isFinite(hz) || hz <= 0) return null;
      const x = (i / sampleRate) * timeScale;
      return `${x},${pitchHzToDisplayY(hz)}`;
    })
    .filter((p): p is string => p != null)
    .join(' ');

export { keys, octaves };
