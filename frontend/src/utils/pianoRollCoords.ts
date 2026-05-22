import { NOTE_HEIGHT } from '../constants/editor';
import { keys, octaves, TOTAL_KEYS } from '../constants/pianoRoll';
import { midiToHz, snapMidi } from './pitch';

const MIDI_MIN = 21;

export const midiToY = (midi: number): number =>
  TOTAL_KEYS * NOTE_HEIGHT - (snapMidi(midi) - MIDI_MIN) * NOTE_HEIGHT + NOTE_HEIGHT / 2;

export const yToMidi = (y: number): number => {
  const midi = TOTAL_KEYS - (y - NOTE_HEIGHT / 2) / NOTE_HEIGHT + MIDI_MIN;
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
  const midi = TOTAL_KEYS - (y - NOTE_HEIGHT / 2) / NOTE_HEIGHT + MIDI_MIN;
  return midiToHz(midi);
};

export const hzToYContinuous = (hz: number): number => {
  const midi = 12 * Math.log2(hz / 440) + 69;
  return TOTAL_KEYS * NOTE_HEIGHT - (midi - MIDI_MIN) * NOTE_HEIGHT + NOTE_HEIGHT / 2;
};

export { keys, octaves };
