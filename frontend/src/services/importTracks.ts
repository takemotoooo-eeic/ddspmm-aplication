import {
  featureToTrack,
  generateDdspAudio,
  parseFluidsynthZip,
  trainDiffusion,
  trainDdsp,
  trainFluidsynth,
} from '../api/backend';
import { AppMode } from '../types/appMode';
import { createOriginalTrack, TrackData } from '../types/trackData';
import { blobDurationSec, signalLengthFromDuration } from '../utils/audio';

const newTrackId = () => `track-${Date.now()}-${Math.random()}`;

export async function importTracksFromFiles(
  mode: AppMode,
  wavFile: File,
  midiFile: File,
): Promise<TrackData[]> {
  const originalTrack = await createOriginalTrack(wavFile);

  if (mode === 'fluidsynth') {
    const zipBlob = await trainFluidsynth(wavFile, midiFile);
    const entries = await parseFluidsynthZip(zipBlob);
    const separated = entries.map(entry => {
      const duration = blobDurationSec(entry.wavBlob);
      return {
        id: newTrackId(),
        name: entry.instrument_name,
        instrument: entry.instrument_name,
        wavData: entry.wavBlob,
        notes: entry.notes,
        signalLength: signalLengthFromDuration(duration),
        muted: false,
        volume: 1,
      };
    });
    return [originalTrack, ...separated];
  }

  const features =
    mode === 'diffusion_ddsp'
      ? await trainDiffusion(wavFile, midiFile)
      : await trainDdsp(wavFile, midiFile);

  const tracks: TrackData[] = [originalTrack];
  for (const feature of features.features) {
    const built = await featureToTrack(feature, generateDdspAudio);
    const notes = built.features.notes ?? [];
    tracks.push({
      id: newTrackId(),
      name: built.name,
      instrument: built.instrument,
      wavData: built.wavData,
      features: { ...built.features, notes },
      notes,
      signalLength: built.signalLength,
      muted: false,
      volume: 1,
    });
  }
  return tracks;
}
