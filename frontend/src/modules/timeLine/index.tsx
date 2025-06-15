import { Box, Typography } from '@mui/material';

interface TimelineProps {
  duration: number; // 秒単位
  width: number;    // ピクセル単位
  height?: number;
}

export const Timeline = ({ duration, width, height = 30 }: TimelineProps) => {
  const interval = 1; // 1秒ごと
  const totalMarkers = Math.ceil(duration / interval);

  return (
    <Box
      sx={{
        height: height,
        bgcolor: '#222',
        borderBottom: '1px solid #333',
        position: 'relative',
        width: width,
        minWidth: width,
        userSelect: 'none',
      }}
    >
      {Array.from({ length: totalMarkers + 1 }).map((_, i) => {
        const left = (i * interval / duration) * width;
        return (
          <Box
            key={i}
            sx={{
              position: 'absolute',
              left,
              top: 0,
              height: '100%',
              width: 0,
              borderLeft: '1px solid #444',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Typography
              variant="caption"
              sx={{
                color: '#888',
                fontSize: '0.7rem',
                position: 'absolute',
                top: (height - 10) / 2,
                left: 2,
                whiteSpace: 'nowrap',
              }}
            >
              {i}s
            </Typography>
          </Box>
        );
      })}
    </Box>
  );
}; 
