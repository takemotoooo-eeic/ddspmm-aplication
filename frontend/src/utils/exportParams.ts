import type { Feature } from '../orval/models/backend-api';
import type { TrackData } from '../types/trackData';

export const tracksToJsonl = (tracks: TrackData[]): string =>
  tracks
    .filter((t): t is TrackData & { features: Feature } => t.features != null)
    .map(t => JSON.stringify(t.features))
    .join('\n');

export const downloadTracksAsJsonl = (
  tracks: TrackData[],
  filename = 'synthesis_params.jsonl',
): void => {
  const lines = tracksToJsonl(tracks);
  if (!lines) return;

  const blob = new Blob([`${lines}\n`], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};
