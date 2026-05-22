import { Box } from '@mui/material';
import { useRef, useState } from 'react';
import {
  LOUDNESS_EDITOR_HEIGHT,
  LOUDNESS_MAX_DB,
  LOUDNESS_MIN_DB,
  PIANO_ROLL_KEY_WIDTH,
  PITCH_SAMPLE_RATE,
  TIME_SCALE,
} from '../../constants/editor';
import { TrackData } from '../../types/trackData';
import { blobDurationSec, durationToWidth } from '../../utils/audio';
import { EditorTimelineRow } from './shared/EditorTimelineRow';
import { PlaybackCursor } from './shared/PlaybackCursor';

interface LoudnessEditorProps {
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
  isEditing: boolean;
  timeScale?: number;
}

const dbToY = (db: number): number => {
  const range = LOUDNESS_MAX_DB - LOUDNESS_MIN_DB;
  return LOUDNESS_EDITOR_HEIGHT - ((db - LOUDNESS_MIN_DB) / range) * LOUDNESS_EDITOR_HEIGHT;
};

const yToDb = (y: number): number => {
  const range = LOUDNESS_MAX_DB - LOUDNESS_MIN_DB;
  const db = LOUDNESS_MIN_DB + ((LOUDNESS_EDITOR_HEIGHT - y) / LOUDNESS_EDITOR_HEIGHT) * range;
  return Math.max(LOUDNESS_MIN_DB, Math.min(LOUDNESS_MAX_DB, db));
};

const DB_TICKS = [-20, -30, -40, -50, -60, -70, -80];

export const LoudnessEditor = ({
  currentTime,
  selectedTrack,
  tracks,
  setTracks,
  setSelectedTrack,
  onTimeLineClick,
  isEditing,
  timeScale = TIME_SCALE,
}: LoudnessEditorProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [tempLoudness, setTempLoudness] = useState<number[] | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const scrollLeftRef = useRef(0);

  const durationSec = tracks.length > 0 ? blobDurationSec(tracks[0].wavData) : 10;
  const contentWidth = durationToWidth(durationSec, timeScale);

  const syncScroll = (scrollLeft: number) => {
    scrollLeftRef.current = scrollLeft;
    if (timelineRef.current) timelineRef.current.scrollLeft = scrollLeft;
    if (editorRef.current) editorRef.current.scrollLeft = scrollLeft;
  };

  const loudnessData = tempLoudness ?? selectedTrack.features?.loudness ?? [];

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditing || !selectedTrack.features) return;
    setIsDragging(true);
    setTempLoudness([...selectedTrack.features.loudness]);
    handleMouseMove(event);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || !tempLoudness || !editorRef.current) return;
    const rect = editorRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left + scrollLeftRef.current;
    const y = event.clientY - rect.top;
    const timeIndex = Math.floor((x / timeScale) * PITCH_SAMPLE_RATE);
    const newDb = yToDb(y);
    if (timeIndex >= 0 && timeIndex < tempLoudness.length) {
      const next = [...tempLoudness];
      next[timeIndex] = newDb;
      setTempLoudness(next);
    }
  };

  const handleMouseUp = () => {
    if (!isDragging || !tempLoudness || !selectedTrack.features) return;
    const updatedTrack: TrackData = {
      ...selectedTrack,
      features: { ...selectedTrack.features, loudness: tempLoudness },
    };
    setTracks(tracks.map(t => (t.id === selectedTrack.id ? updatedTrack : t)));
    setSelectedTrack(updatedTrack);
    setIsDragging(false);
    setTempLoudness(null);
  };

  const points = loudnessData
    .map((db, index) => {
      const x = (index / PITCH_SAMPLE_RATE) * timeScale;
      const y = dbToY(db);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <EditorTimelineRow
        durationSec={durationSec}
        contentWidth={contentWidth}
        currentTime={currentTime}
        timeScale={timeScale}
        onTimeLineClick={onTimeLineClick}
        scrollRef={timelineRef}
        onScroll={syncScroll}
      />
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <Box
          sx={{
            width: PIANO_ROLL_KEY_WIDTH,
            bgcolor: '#222',
            borderRight: '1px solid #333',
            height: LOUDNESS_EDITOR_HEIGHT,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            py: 1,
          }}
        >
          {DB_TICKS.map(db => (
            <Box key={db} sx={{ color: '#fff', fontSize: 12, textAlign: 'right', pr: 1 }}>
              {db}
            </Box>
          ))}
        </Box>
        <Box
          ref={editorRef}
          sx={{
            flex: 1,
            bgcolor: '#181818',
            position: 'relative',
            overflowX: 'auto',
            overflowY: 'hidden',
            height: LOUDNESS_EDITOR_HEIGHT,
            cursor: isEditing ? 'crosshair' : 'default',
          }}
          onScroll={e => syncScroll(e.currentTarget.scrollLeft)}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <Box sx={{ width: contentWidth, height: LOUDNESS_EDITOR_HEIGHT, position: 'relative' }}>
            <PlaybackCursor
              currentTime={currentTime}
              durationSec={durationSec}
              contentWidth={contentWidth}
            />
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
            >
              <polyline points={points} fill="none" stroke="#646cff" strokeWidth={2} />
            </svg>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
