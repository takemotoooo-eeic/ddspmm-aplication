import { BYTES_PER_SAMPLE, SAMPLE_RATE } from '../constants/editor';

const WAV_HEADER_BYTES = 44;

const readFourCC = (buf: Uint8Array, offset: number): string =>
  String.fromCharCode(buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]);

/** WAV ヘッダから再生長（秒）を取得。非標準フォーマットの場合は null。 */
export const parseWavDurationSec = (buf: Uint8Array): number | null => {
  if (buf.length < 44) return null;
  if (readFourCC(buf, 0) !== 'RIFF' || readFourCC(buf, 8) !== 'WAVE') return null;

  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let offset = 12;
  let sampleRate = SAMPLE_RATE;
  let numChannels = 1;
  let bitsPerSample = 16;
  let dataSize = 0;

  while (offset + 8 <= buf.length) {
    const chunkId = readFourCC(buf, offset);
    const chunkSize = view.getUint32(offset + 4, true);
    const dataStart = offset + 8;

    if (chunkId === 'fmt ' && dataStart + 16 <= buf.length) {
      numChannels = view.getUint16(dataStart + 2, true);
      sampleRate = view.getUint32(dataStart + 4, true);
      bitsPerSample = view.getUint16(dataStart + 14, true);
    } else if (chunkId === 'data') {
      dataSize = chunkSize;
    }

    offset = dataStart + chunkSize + (chunkSize % 2);
  }

  if (dataSize <= 0) return null;
  const bytesPerFrame = numChannels * (bitsPerSample / 8);
  if (bytesPerFrame <= 0 || sampleRate <= 0) return null;
  return dataSize / (sampleRate * bytesPerFrame);
};

export async function wavDurationSec(wavBlob: Blob): Promise<number> {
  const buf = new Uint8Array(await wavBlob.slice(0, 65536).arrayBuffer());
  return parseWavDurationSec(buf) ?? blobDurationSec(wavBlob);
}

/** バックエンド生成 PCM WAV（16kHz mono）の再生長（秒）概算。 */
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
