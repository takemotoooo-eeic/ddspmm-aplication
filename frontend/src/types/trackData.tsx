import { Feature, Note } from '../orval/models/backend-api';
import { blobDurationSec, wavDurationSec } from '../utils/audio';
import { AppMode } from './appMode';

export const ORIGINAL_TRACK_NAME = 'original';
export const EDITED_TRACK_NAME = 'edited';

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
  /** WAV ヘッダから取得した再生長（秒）。original 等の非 16kHz mono 向け */
  durationSec?: number;
  muted: boolean;
  volume: number;
  /** 入力参照混合音トラック（編集不可） */
  isOriginal?: boolean;
}

export const isOriginalTrack = (track: TrackData): boolean =>
  track.isOriginal === true || track.name === ORIGINAL_TRACK_NAME;

export const isEditedTrack = (track: TrackData): boolean =>
  track.name === EDITED_TRACK_NAME;

export const trackDurationSec = (track: TrackData): number =>
  track.durationSec ?? blobDurationSec(track.wavData);

export const createOriginalTrack = async (wavBlob: Blob): Promise<TrackData> => ({
  id: 'original',
  name: ORIGINAL_TRACK_NAME,
  instrument: ORIGINAL_TRACK_NAME,
  wavData: wavBlob,
  durationSec: await wavDurationSec(wavBlob),
  muted: true,
  volume: 1,
  isOriginal: true,
});

export const createEditedTrack = async (wavBlob: Blob): Promise<TrackData> => ({
  id: 'edited',
  name: EDITED_TRACK_NAME,
  instrument: EDITED_TRACK_NAME,
  wavData: wavBlob,
  durationSec: await wavDurationSec(wavBlob),
  muted: false,
  volume: 1,
});

export const trackNotes = (track: TrackData): Note[] =>
  track.features?.notes ?? track.notes ?? [];

export const supportsParamLoad = (mode: AppMode): boolean =>
  mode === 'diffusion_ddsp' || mode === 'ddsp';

