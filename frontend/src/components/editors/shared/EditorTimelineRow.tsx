import { Box } from '@mui/material';
import { useRef } from 'react';
import { PIANO_ROLL_KEY_WIDTH, TIMELINE_HEIGHT } from '../../../constants/editor';
import { Timeline } from '../../../modules/timeLine';
import { PlaybackCursor } from './PlaybackCursor';

interface EditorTimelineRowProps {
  durationSec: number;
  contentWidth: number;
  currentTime: number;
  timeScale: number;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
  scrollRef?: React.RefObject<HTMLDivElement | null>;
  onScroll?: (scrollLeft: number) => void;
}

export const EditorTimelineRow = ({
  durationSec,
  contentWidth,
  currentTime,
  onTimeLineClick,
  scrollRef,
  onScroll,
}: EditorTimelineRowProps) => {
  const internalRef = useRef<HTMLDivElement>(null);
  const ref = scrollRef ?? internalRef;

  return (
    <Box sx={{ position: 'relative', height: TIMELINE_HEIGHT, display: 'flex', width: '100%' }}>
      <Box
        sx={{
          width: PIANO_ROLL_KEY_WIDTH,
          bgcolor: '#222',
          borderRight: '1px solid #333',
          height: '100%',
          flexShrink: 0,
        }}
      />
      <Box
        ref={ref}
        sx={{
          flex: 1,
          position: 'relative',
          overflowX: 'auto',
          overflowY: 'hidden',
        }}
        onScroll={e => onScroll?.(e.currentTarget.scrollLeft)}
      >
        <Box
          sx={{ width: contentWidth, height: '100%', position: 'relative' }}
          onClick={onTimeLineClick}
        >
          <PlaybackCursor
            currentTime={currentTime}
            durationSec={durationSec}
            contentWidth={contentWidth}
          />
          <Timeline duration={durationSec || 10} width={contentWidth} height={TIMELINE_HEIGHT} />
        </Box>
      </Box>
    </Box>
  );
};
