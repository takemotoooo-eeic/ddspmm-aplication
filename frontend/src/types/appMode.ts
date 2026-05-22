export type AppMode = 'diffusion_ddsp' | 'ddsp' | 'fluidsynth';

export const APP_MODE_LABELS: Record<AppMode, string> = {
  diffusion_ddsp: 'Diffusion+DDSPMM',
  ddsp: 'DDSPMM',
  fluidsynth: 'FluidSynth',
};

export const APP_MODES: AppMode[] = ['diffusion_ddsp', 'ddsp', 'fluidsynth'];
