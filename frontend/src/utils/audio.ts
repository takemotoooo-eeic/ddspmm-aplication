import { BYTES_PER_SAMPLE, SAMPLE_RATE } from '../constants/editor';

export const blobDurationSec = (wavBlob: Blob): number =>
  wavBlob.size / (SAMPLE_RATE * BYTES_PER_SAMPLE);

export const durationToWidth = (durationSec: number, timeScale: number): number =>
  Math.floor(durationSec * timeScale);

export const signalLengthFromDuration = (durationSec: number): number =>
  Math.floor(durationSec * SAMPLE_RATE);
