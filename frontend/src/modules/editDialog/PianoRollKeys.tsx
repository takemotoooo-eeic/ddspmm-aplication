import { Box, Typography } from '@mui/material';

const keys = [
  { note: 'B', isBlack: false },
  { note: 'A#', isBlack: true },
  { note: 'A', isBlack: false },
  { note: 'G#', isBlack: true },
  { note: 'G', isBlack: false },
  { note: 'F#', isBlack: true },
  { note: 'F', isBlack: false },
  { note: 'E', isBlack: false },
  { note: 'D#', isBlack: true },
  { note: 'D', isBlack: false },
  { note: 'C#', isBlack: true },
  { note: 'C', isBlack: false },
];

const octaves = [5, 4, 3, 2]; // 表示したいオクターブ

export const PianoRollKeys = () => (
  <Box sx={{ width: 75, height: '100%', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
    <Box sx={{ height: 20, flexShrink: 0, bgcolor: 'transparent' }} />
    {octaves.map((oct) =>
      keys.map((key) => (
        <Box
          key={`${key.note}${oct}`}
          sx={{
            height: 30,
            flexShrink: 0,
            bgcolor: key.isBlack ? '#222' : '#fff',
            border: '1px solid #333',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
          }}
        >
          {/* Cの時だけラベル表示 */}
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
      ))
    )}
  </Box>
); 
