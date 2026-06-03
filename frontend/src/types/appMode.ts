export type AppMode = 'diffusion_ddsp' | 'ddsp' | 'fluidsynth' | 'ttm';

export const APP_MODE_LABELS: Record<AppMode, string> = {
  diffusion_ddsp: 'Diffusion+DDSPMM',
  ddsp: 'DDSPMM',
  fluidsynth: 'FluidSynth',
  ttm: 'TTM',
};

export const APP_MODES: AppMode[] = ['diffusion_ddsp', 'ddsp', 'fluidsynth', 'ttm'];
