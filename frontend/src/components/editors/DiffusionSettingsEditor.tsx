import {
  Box,
  FormControl,
  MenuItem,
  Select,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';

const DEFAULT_DENOISING_STEPS = 1000;
const DEFAULT_USE_DDIM_VALUE = true;
export const DENOISING_STEP_OPTIONS = [10, 50, 100, 200, 500, 1000] as const;

interface DiffusionSettingsEditorProps {
  numDenoisingSteps: number;
  onNumDenoisingStepsChange: (value: number) => void;
  useDdim: boolean;
  onUseDdimChange: (value: boolean) => void;
}

export const DEFAULT_NUM_DENOISING_STEPS = DEFAULT_DENOISING_STEPS;
export const DEFAULT_USE_DDIM = DEFAULT_USE_DDIM_VALUE;

export const DiffusionSettingsEditor = ({
  numDenoisingSteps,
  onNumDenoisingStepsChange,
  useDdim,
  onUseDdimChange,
}: DiffusionSettingsEditorProps) => {
  const handleSliderChange = (_: Event, value: number | number[]) => {
    onNumDenoisingStepsChange(DENOISING_STEP_OPTIONS[value as number]);
  };
  const selectedStepIndex = Math.max(
    0,
    DENOISING_STEP_OPTIONS.findIndex(value => value === numDenoisingSteps),
  );

  return (
    <Box sx={{ color: '#fff' }}>
      <Typography variant="body2" sx={{ mb: 3, color: '#aaa' }}>
        Number of timesteps used for REGENERATE and for /diffusion/generate when
        editing notes.
      </Typography>

      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Sampler
      </Typography>
      <ToggleButtonGroup
        value={useDdim ? 'ddim' : 'ddpm'}
        exclusive
        onChange={(_, value: 'ddim' | 'ddpm' | null) => {
          if (value) onUseDdimChange(value === 'ddim');
        }}
        size="small"
        sx={{ mb: 3 }}
      >
        <ToggleButton value="ddim">DDIM</ToggleButton>
        <ToggleButton value="ddpm">DDPM</ToggleButton>
      </ToggleButtonGroup>

      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Timesteps
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, maxWidth: 480 }}>
        <Slider
          value={selectedStepIndex}
          min={0}
          max={DENOISING_STEP_OPTIONS.length - 1}
          step={1}
          marks={DENOISING_STEP_OPTIONS.map((value, index) => ({
            value: index,
            label: String(value),
          }))}
          onChange={handleSliderChange}
          sx={{ flex: 1 }}
        />
        <FormControl size="small" sx={{ width: 120 }}>
          <Select
            value={numDenoisingSteps}
            onChange={e => onNumDenoisingStepsChange(Number(e.target.value))}
            sx={{ color: '#fff', bgcolor: '#333' }}
          >
            {DENOISING_STEP_OPTIONS.map(value => (
              <MenuItem key={value} value={value}>
                {value}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: '#888' }}>
        Choose one of 10, 50, 100, 200, 500, or 1000 steps.
      </Typography>
    </Box>
  );
};
