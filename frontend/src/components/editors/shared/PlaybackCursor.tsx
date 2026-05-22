import { Box } from '@mui/material';

interface PlaybackCursorProps {
  currentTime: number;
  durationSec: number;
  contentWidth: number;
}

export const PlaybackCursor = ({
  currentTime,
  durationSec,
  contentWidth,
}: PlaybackCursorProps) => (
  <Box
    sx={{
      position: 'absolute',
      left: durationSec > 0 ? (currentTime / durationSec) * contentWidth : 0,
      top: 0,
      height: '100%',
      width: 2,
      bgcolor: 'rgba(255, 255, 255, 0.3)',
      zIndex: 2,
      pointerEvents: 'none',
    }}
  />
);
