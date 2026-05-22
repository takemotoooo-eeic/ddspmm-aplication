import { Feature, Note } from '../orval/models/backend-api';
import { AppMode } from './appMode';

export interface TrackData {
  id: string;
  name: string;
  instrument: string;
  wavData: Blob;
  /** DDSP / Diffusion モード用 */
  features?: Feature;
  /** FluidSynth モード用（features が無い場合） */
  notes?: Note[];
  signalLength?: number;
  muted: boolean;
  volume: number;
}

export const trackNotes = (track: TrackData): Note[] =>
  track.features?.notes ?? track.notes ?? [];

export const supportsParamLoad = (mode: AppMode): boolean =>
  mode === 'diffusion_ddsp' || mode === 'ddsp';

