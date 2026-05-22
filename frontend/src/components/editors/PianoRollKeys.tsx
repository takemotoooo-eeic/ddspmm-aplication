import { Box, Typography } from '@mui/material';
import { NOTE_HEIGHT } from '../../constants/editor';
import { keys, octaves } from '../../constants/pianoRoll';

export const PianoRollKeys = () => (
  <Box sx={{ width: 75, display: 'flex', flexDirection: 'column' }}>
    {octaves.map(oct =>
      keys.map(key => (
        <Box
          key={`${key.note}${oct}`}
          sx={{
            height: NOTE_HEIGHT,
            flexShrink: 0,
            bgcolor: key.isBlack ? '#222' : '#fff',
            border: '1px solid #333',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
          }}
        >
          {key.note === 'C' && (
            <Typography
              variant="caption"
              sx={{
                color: '#000',
                position: 'absolute',
                left: 24,
                top: 4,
                fontWeight: 'bold',
                fontSize: '1rem',
              }}
            >
              {`${key.note}${oct}`}
            </Typography>
          )}
        </Box>
      )),
    )}
  </Box>
);
