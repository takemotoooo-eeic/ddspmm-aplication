import { instance } from '../libs/mutator';
import type {
  DDSPGenerateParams,
  DiffusionGenerateParams as OrvalDiffusionGenerateParams,
  Feature,
  Features,
  Note,
} from '../orval/models/backend-api';
import { unzipToMap } from '../utils/zip';

export type DiffusionGenerateParams = OrvalDiffusionGenerateParams & {
  num_denoising_steps?: number | null;
  use_ddim?: boolean | null;
  note_operation?: NoteOperation | null;
  operation_prev_note?: Note | null;
  operation_note?: Note | null;
  prev_features?: DDSPGenerateParams | null;
};

export type NoteOperation = 'add' | 'delete' | 'move' | 'resize';

export interface NoteOperationPayload {
  operation_prev_note?: Note | null;
  operation_note?: Note | null;
}

export interface FluidsynthGenerateParams {
  notes: Note[];
  instrument_name: string;
  signal_length: number;
}

export interface FluidsynthZipEntry {
  instrument_name: string;
  wavBlob: Blob;
  notes: Note[];
}

export async function trainDiffusion(wavFile: File, midiFile: File): Promise<Features> {
  const form = new FormData();
  form.append('wav_file', wavFile);
  form.append('midi_file', midiFile);
  const { data } = await instance.post<Features>('/diffusion/train', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function trainDdsp(wavFile: File, midiFile: File): Promise<Features> {
  const form = new FormData();
  form.append('wav_file', wavFile);
  form.append('midi_file', midiFile);
  const { data } = await instance.post<Features>('/ddsp/train', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function trainFluidsynth(wavFile: File, midiFile: File): Promise<Blob> {
  const form = new FormData();
  form.append('wav_file', wavFile);
  form.append('midi_file', midiFile);
  const { data } = await instance.post<Blob>('/fluidsynth/train', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  });
  return data;
}

export async function generateDdspAudio(params: DDSPGenerateParams): Promise<Blob> {
  const { data } = await instance.post<Blob>('/ddsp/generate', params, {
    responseType: 'blob',
  });
  return data;
}

export async function generateDiffusionParams(
  params: DiffusionGenerateParams,
): Promise<DDSPGenerateParams> {
  const { data } = await instance.post<DDSPGenerateParams>('/diffusion/generate', params);
  return data;
}

export interface MelodyflowEditParams {
  wavFile: Blob;
  startSec: number;
  endSec: number;
  text: string;
}

export async function editMelodyflow(params: MelodyflowEditParams): Promise<Blob> {
  const form = new FormData();
  form.append('wav_file', params.wavFile, 'audio.wav');
  form.append('start_sec', String(params.startSec));
  form.append('end_sec', String(params.endSec));
  form.append('text', params.text);
  const { data } = await instance.post<Blob>('/melodyflow/edit', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  });
  return data;
}

export async function generateFluidsynthAudio(params: FluidsynthGenerateParams): Promise<Blob> {
  const { data } = await instance.post<Blob>('/fluidsynth/generate', params, {
    responseType: 'blob',
  });
  return data;
}

export async function parseFluidsynthZip(zipBlob: Blob): Promise<FluidsynthZipEntry[]> {
  const files = await unzipToMap(zipBlob);
  const manifestBlob = files.get('manifest.json');
  if (!manifestBlob) {
    throw new Error('manifest.json not found in FluidSynth ZIP');
  }
  const manifest = JSON.parse(await manifestBlob.text()) as {
    features: Array<{ instrument_name: string; wav_file?: string; notes: Note[] }>;
  };

  const entries: FluidsynthZipEntry[] = [];
  for (const feature of manifest.features) {
    const wavKey =
      feature.wav_file ??
      `${feature.instrument_name.replace(/[^a-zA-Z0-9_-]/g, '_')}.wav`;
    const wavBlob = files.get(wavKey);
    if (!wavBlob) {
      throw new Error(
        `WAV not found for instrument: ${feature.instrument_name} (expected: ${wavKey})`,
      );
    }
    entries.push({
      instrument_name: feature.instrument_name,
      wavBlob: new Blob([await wavBlob.arrayBuffer()], { type: 'audio/wav' }),
      notes: feature.notes,
    });
  }
  return entries;
}

export async function featureToTrack(
  feature: Feature,
  generateWav: (params: DDSPGenerateParams) => Promise<Blob>,
): Promise<{
  name: string;
  instrument: string;
  wavData: Blob;
  features: Feature;
  signalLength: number;
}> {
  const body: DDSPGenerateParams = {
    z_feature: feature.z_feature,
    loudness: feature.loudness,
    pitch: feature.pitch,
  };
  const wavData = await generateWav(body);
  const blockSize = 512;
  const signalLength = feature.pitch.length * blockSize;
  const notes = feature.notes ?? [];
  return {
    name: feature.instrument_name,
    instrument: feature.instrument_name,
    wavData,
    features: { ...feature, notes },
    signalLength,
  };
}
