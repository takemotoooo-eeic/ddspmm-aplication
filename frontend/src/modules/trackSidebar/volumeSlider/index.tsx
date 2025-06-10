import { Box, Slider } from '@mui/material';
import { useState } from 'react';

interface VolumeSliderProps {
  value: number;
  onChange: (value: number) => void;
}

export const VolumeSlider = ({ value, onChange }: VolumeSliderProps) => {
  const [localValue, setLocalValue] = useState(value);

  const handleChange = (_event: Event, newValue: number | number[]) => {
    const volume = newValue as number;
    setLocalValue(volume);
    onChange(volume);
  };

  return (
    <Box sx={{ width: 80, display: 'flex', alignItems: 'center', gap: 1 }}>
      <Slider
        size="small"
        min={0}
        max={128}
        value={localValue}
        onChange={handleChange}
        sx={{
          color: '#646cff',
          '& .MuiSlider-thumb': {
            width: 12,
            height: 12,
          },
        }}
      />
    </Box>
  );
}; 
