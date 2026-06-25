import { Box, CircularProgress } from '@mui/material';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  LOUDNESS_MAX_DB,
  LOUDNESS_MIN_DB,
  PIANO_ROLL_KEY_WIDTH,
  PITCH_SAMPLE_RATE,
  SAMPLE_RATE,
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
  isBusy?: boolean;
  initialScrollLeft?: number;
  onScrollLeftChange?: (scrollLeft: number) => void;
  onLoudnessChange?: (loudness: number[]) => void;
}

const DEFAULT_EDITOR_HEIGHT = 480;

const dbToY = (db: number, height: number): number => {
  const range = LOUDNESS_MAX_DB - LOUDNESS_MIN_DB;
  const clamped = Math.max(LOUDNESS_MIN_DB, Math.min(LOUDNESS_MAX_DB, db));
  return height - ((clamped - LOUDNESS_MIN_DB) / range) * height;
};

const yToDb = (y: number, height: number): number => {
  const range = LOUDNESS_MAX_DB - LOUDNESS_MIN_DB;
  const db = LOUDNESS_MIN_DB + ((height - y) / height) * range;
  return Math.max(LOUDNESS_MIN_DB, Math.min(LOUDNESS_MAX_DB, db));
};

const DB_TICKS = [0, -20, -40, -60, -80, -100];

export const LoudnessEditor = ({
  currentTime,
  selectedTrack,
  tracks,
  setTracks,
  setSelectedTrack,
  onTimeLineClick,
  isEditing,
  timeScale = TIME_SCALE,
  isBusy = false,
  initialScrollLeft,
  onScrollLeftChange,
  onLoudnessChange,
}: LoudnessEditorProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [tempLoudness, setTempLoudness] = useState<number[] | null>(null);
  const [editorHeight, setEditorHeight] = useState(DEFAULT_EDITOR_HEIGHT);
  const timelineRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const scrollLeftRef = useRef(0);

  const loudnessData = tempLoudness ?? selectedTrack.features?.loudness ?? [];
  const durationSec =
    selectedTrack.signalLength != null
      ? selectedTrack.signalLength / SAMPLE_RATE
      : loudnessData.length > 0
        ? loudnessData.length / PITCH_SAMPLE_RATE
        : blobDurationSec(selectedTrack.wavData);
  const contentWidth = durationToWidth(durationSec, timeScale);

  const syncScroll = useCallback((scrollLeft: number) => {
    scrollLeftRef.current = scrollLeft;
    if (timelineRef.current) timelineRef.current.scrollLeft = scrollLeft;
    if (editorRef.current) editorRef.current.scrollLeft = scrollLeft;
    onScrollLeftChange?.(scrollLeft);
  }, [onScrollLeftChange]);

  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;

    const updateHeight = () => {
      if (el.clientHeight > 0) setEditorHeight(el.clientHeight);
    };
    updateHeight();

    const resizeObserver = new ResizeObserver(updateHeight);
    resizeObserver.observe(el);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    if (initialScrollLeft == null) return;
    const frame = requestAnimationFrame(() => syncScroll(initialScrollLeft));
    return () => cancelAnimationFrame(frame);
  }, [initialScrollLeft, selectedTrack.id, syncScroll]);

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditing || isBusy || !selectedTrack.features) return;
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
    const newDb = yToDb(y, rect.height || editorHeight);
    if (timeIndex >= 0 && timeIndex < tempLoudness.length) {
      const next = [...tempLoudness];
      next[timeIndex] = newDb;
      setTempLoudness(next);
    }
  };

  const handleMouseUp = () => {
    if (!isDragging || !tempLoudness || !selectedTrack.features) return;
    if (onLoudnessChange) {
      onLoudnessChange(tempLoudness);
      setIsDragging(false);
      setTempLoudness(null);
      return;
    }
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
      if (!Number.isFinite(db)) return null;
      const x = (index / PITCH_SAMPLE_RATE) * timeScale;
      return `${x},${dbToY(db, editorHeight)}`;
    })
    .filter((p): p is string => p != null)
    .join(' ');

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {isBusy && (
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            bgcolor: 'rgba(0,0,0,0.35)',
            zIndex: 30,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <CircularProgress size={32} sx={{ color: '#fff' }} />
        </Box>
      )}
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
            height: '100%',
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
            height: '100%',
            cursor: isBusy ? 'wait' : isEditing ? 'crosshair' : 'default',
            opacity: isBusy ? 0.6 : 1,
            pointerEvents: isBusy ? 'none' : 'auto',
          }}
          onScroll={e => syncScroll(e.currentTarget.scrollLeft)}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <Box sx={{ width: contentWidth, height: '100%', position: 'relative' }}>
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
              {points.length > 0 && (
                <polyline points={points} fill="none" stroke="#646cff" strokeWidth={3} />
              )}
            </svg>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
