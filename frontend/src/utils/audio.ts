import { BYTES_PER_SAMPLE, SAMPLE_RATE } from '../constants/editor';

const WAV_HEADER_BYTES = 44;

/** PCM WAV の再生長（秒）。ヘッダ分を除いたデータサイズから概算。 */
export const blobDurationSec = (wavBlob: Blob): number => {
  if (!wavBlob || wavBlob.size <= WAV_HEADER_BYTES) return 0;
  return (wavBlob.size - WAV_HEADER_BYTES) / (SAMPLE_RATE * BYTES_PER_SAMPLE);
};

export const durationToWidth = (durationSec: number, timeScale: number): number =>
  Math.floor(durationSec * timeScale);

export const signalLengthFromDuration = (durationSec: number): number =>
  Math.floor(durationSec * SAMPLE_RATE);

export const safeAudioFilename = (name: string): string =>
  name.replace(/[^a-zA-Z0-9_-]/g, '_') || 'track';

export const downloadWavBlob = (wavBlob: Blob, filename: string): void => {
  const url = URL.createObjectURL(wavBlob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename.endsWith('.wav') ? filename : `${filename}.wav`;
  anchor.click();
  URL.revokeObjectURL(url);
};
