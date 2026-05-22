import { ToggleButton, ToggleButtonGroup } from '@mui/material';
import { APP_MODE_LABELS, APP_MODES, AppMode } from '../../types/appMode';

interface ModeSelectorProps {
  mode: AppMode;
  onChange: (mode: AppMode) => void;
}

export const ModeSelector = ({ mode, onChange }: ModeSelectorProps) => (
  <ToggleButtonGroup
    value={mode}
    exclusive
    onChange={(_, value: AppMode | null) => {
      if (value) onChange(value);
    }}
    size="small"
    sx={{
      mr: 2,
      '& .MuiToggleButton-root': {
        color: '#fff',
        borderColor: '#444',
        fontSize: '0.75rem',
        px: 1.5,
        py: 0.5,
        '&.Mui-selected': {
          bgcolor: '#646cff',
          color: '#fff',
          '&:hover': { bgcolor: '#535bf2' },
        },
      },
    }}
  >
    {APP_MODES.map(m => (
      <ToggleButton key={m} value={m}>
        {APP_MODE_LABELS[m]}
      </ToggleButton>
    ))}
  </ToggleButtonGroup>
);
