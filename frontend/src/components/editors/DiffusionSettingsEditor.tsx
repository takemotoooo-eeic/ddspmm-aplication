import { Box, Slider, TextField, Typography } from '@mui/material';

const MIN_DENOISING_STEPS = 1;
const MAX_DENOISING_STEPS = 1000;
const DEFAULT_DENOISING_STEPS = 1000;

interface DiffusionSettingsEditorProps {
  numDenoisingSteps: number;
  onNumDenoisingStepsChange: (value: number) => void;
}

export const DEFAULT_NUM_DENOISING_STEPS = DEFAULT_DENOISING_STEPS;

export const DiffusionSettingsEditor = ({
  numDenoisingSteps,
  onNumDenoisingStepsChange,
}: DiffusionSettingsEditorProps) => {
  const handleSliderChange = (_: Event, value: number | number[]) => {
    onNumDenoisingStepsChange(value as number);
  };

  const handleInputChange = (raw: string) => {
    const parsed = Number.parseInt(raw, 10);
    if (Number.isNaN(parsed)) return;
    onNumDenoisingStepsChange(
      Math.min(MAX_DENOISING_STEPS, Math.max(MIN_DENOISING_STEPS, parsed)),
    );
  };

  return (
    <Box sx={{ color: '#fff' }}>
      <Typography variant="body2" sx={{ mb: 3, color: '#aaa' }}>
        REGENERATE およびノート編集時の /diffusion/generate
        リクエストで使用するタイムステップ数を設定します。
      </Typography>

      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Timesteps
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, maxWidth: 480 }}>
        <Slider
          value={numDenoisingSteps}
          min={MIN_DENOISING_STEPS}
          max={MAX_DENOISING_STEPS}
          step={1}
          onChange={handleSliderChange}
          sx={{ flex: 1 }}
        />
        <TextField
          type="number"
          value={numDenoisingSteps}
          onChange={e => handleInputChange(e.target.value)}
          inputProps={{
            min: MIN_DENOISING_STEPS,
            max: MAX_DENOISING_STEPS,
            step: 1,
          }}
          size="small"
          sx={{
            width: 100,
            input: { color: '#fff' },
            '& .MuiOutlinedInput-root': { bgcolor: '#333' },
          }}
        />
      </Box>
      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: '#888' }}>
        1〜1000（デフォルト: 1000）。1000 未満の場合は DDIM サンプリングを使用します。
      </Typography>
    </Box>
  );
};
