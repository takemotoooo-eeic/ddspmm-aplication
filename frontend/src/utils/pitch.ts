/** A4 = 440 Hz, MIDI 69 */
const A4_HZ = 440;
const A4_MIDI = 69;

export const hzToMidi = (hz: number): number =>
  12 * Math.log2(hz / A4_HZ) + A4_MIDI;

export const midiToHz = (midi: number): number =>
  A4_HZ * Math.pow(2, (midi - A4_MIDI) / 12);

/** 半音グリッドにスナップした周波数 */
export const snapHzToSemitone = (hz: number): number =>
  midiToHz(Math.round(hzToMidi(hz)));

export const snapMidi = (midi: number): number => Math.round(midi);
