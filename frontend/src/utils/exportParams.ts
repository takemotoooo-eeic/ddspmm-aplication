import type { DDSPGenerateParams, Feature } from '../orval/models/backend-api';
import {
  createOriginalTrack,
  isOriginalTrack,
  type TrackData,
} from '../types/trackData';
import { zipFiles, unzipToMap } from './zip';

export const EXPORT_ORIGINAL_WAV = 'original.wav';
export const EXPORT_PARAMS_JSONL = 'synthesis_params.jsonl';
export const EXPORT_FILENAME = 'tracks_export.zip';

export const tracksToJsonl = (tracks: TrackData[]): string =>
  tracks
    .filter(
      (t): t is TrackData & { features: Feature } =>
        t.features != null && !isOriginalTrack(t),
    )
    .map(t => JSON.stringify(t.features))
    .join('\n');

export const downloadTracksExport = async (
  tracks: TrackData[],
  filename = EXPORT_FILENAME,
): Promise<void> => {
  const jsonl = tracksToJsonl(tracks);
  const original = tracks.find(isOriginalTrack);
  if (!jsonl || !original?.wavData) return;

  const zipBlob = await zipFiles([
    { name: EXPORT_ORIGINAL_WAV, blob: original.wavData },
    {
      name: EXPORT_PARAMS_JSONL,
      blob: new Blob([`${jsonl}\n`], { type: 'application/x-ndjson' }),
    },
  ]);

  const url = URL.createObjectURL(zipBlob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

export async function loadTracksFromExportFile(
  file: File,
  generateWav: (params: DDSPGenerateParams) => Promise<Blob>,
): Promise<TrackData[]> {
  if (!file.name.toLowerCase().endsWith('.zip')) {
    throw new Error('Only ZIP files can be loaded');
  }

  const files = await unzipToMap(file);
  const jsonlBlob = files.get(EXPORT_PARAMS_JSONL);
  const originalWav = files.get(EXPORT_ORIGINAL_WAV);

  if (!jsonlBlob) {
    throw new Error(`${EXPORT_PARAMS_JSONL} not found in ZIP`);
  }
  if (!originalWav || originalWav.size === 0) {
    throw new Error(`${EXPORT_ORIGINAL_WAV} not found in ZIP`);
  }

  const tracks: TrackData[] = [await createOriginalTrack(originalWav)];
  const jsonlText = await jsonlBlob.text();

  const lines = jsonlText.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const data = JSON.parse(line);
    const body: DDSPGenerateParams = {
      z_feature: data.z_feature,
      loudness: data.loudness,
      pitch: data.pitch,
    };
    const wav = await generateWav(body);
    const blockSize = 512;
    const notes = data.notes ?? [];
    tracks.push({
      id: `track-${Date.now()}-${Math.random()}`,
      name: data.instrument_name,
      instrument: data.instrument_name,
      wavData: wav,
      features: { ...data, notes },
      notes,
      signalLength: data.pitch.length * blockSize,
      muted: false,
      volume: 1,
    });
  }

  return tracks;
}
