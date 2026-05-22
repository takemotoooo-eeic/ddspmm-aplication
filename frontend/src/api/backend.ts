import { instance } from '../libs/mutator';
import { unzipToMap } from '../utils/zip';
import type {
  DDSPGenerateParams,
  DiffusionGenerateParams,
  Feature,
  Features,
  Note,
} from '../orval/models/backend-api';

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
    features: Array<{ instrument_name: string; notes: Note[] }>;
  };

  const entries: FluidsynthZipEntry[] = [];
  for (const feature of manifest.features) {
    const safeName = feature.instrument_name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const wavBlob = files.get(`${safeName}.wav`);
    if (!wavBlob) {
      throw new Error(`WAV not found for instrument: ${feature.instrument_name}`);
    }
    entries.push({
      instrument_name: feature.instrument_name,
      wavBlob,
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
  return {
    name: feature.instrument_name,
    instrument: feature.instrument_name,
    wavData,
    features: feature,
    signalLength,
  };
}
