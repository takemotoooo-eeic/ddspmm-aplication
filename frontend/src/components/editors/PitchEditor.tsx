import { Box, CircularProgress } from '@mui/material';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  NOTE_HEIGHT,
  PIANO_ROLL_KEY_WIDTH,
  PITCH_SAMPLE_RATE,
  SAMPLE_RATE,
  TIME_SCALE,
} from '../../constants/editor';
import { keys, octaves, PIANO_ROLL_HEIGHT } from '../../constants/pianoRoll';
import { usePianoRollScroll } from '../../hooks/usePianoRollScroll';
import type { Note } from '../../orval/models/backend-api';
import { TrackData, trackNotes } from '../../types/trackData';
import { blobDurationSec, durationToWidth } from '../../utils/audio';
import {
  buildPitchPolylinePoints,
  noteFrequencyToRectY,
  pitchHzToDisplayY,
  yToHz,
  yToHzContinuous,
} from '../../utils/pianoRollCoords';
import { PianoRollKeys } from './PianoRollKeys';
import { EditorTimelineRow } from './shared/EditorTimelineRow';
import { PlaybackCursor } from './shared/PlaybackCursor';

interface PitchEditorProps {
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
  isEditing: boolean;
  timeScale?: number;
  /** Edit オフ時: ノートを半音単位でドラッグ可能 */
  enableNoteDrag?: boolean;
  /** ノートドロップ後に diffusion/generate → feature 更新 */
  onNoteDrop?: (notes: Note[]) => Promise<void>;
  isBusy?: boolean;
}

export const PitchEditor = ({
  currentTime,
  selectedTrack,
  tracks,
  setTracks,
  setSelectedTrack,
  onTimeLineClick,
  isEditing,
  timeScale = TIME_SCALE,
  enableNoteDrag = false,
  onNoteDrop,
  isBusy = false,
}: PitchEditorProps) => {
  const [isDraggingPitch, setIsDraggingPitch] = useState(false);
  const [tempPitch, setTempPitch] = useState<number[] | null>(null);
  const [draggedNoteIndex, setDraggedNoteIndex] = useState<number | null>(null);
  const [tempNotes, setTempNotes] = useState<Note[] | null>(null);
  const noteGrabOffsetXRef = useRef(0);
  const {
    timelineRef,
    pianoRollRef,
    keysRef,
    scrollLeftRef,
    scrollTopRef,
    syncScrollLeft,
    syncScrollTopTo,
    handlePianoRollScroll,
    handleKeysScroll,
  } = usePianoRollScroll();

  const pitchForScroll = selectedTrack.features?.pitch ?? [];
  const pitchData = tempPitch ?? pitchForScroll;
  const durationSec =
    selectedTrack.signalLength != null
      ? selectedTrack.signalLength / SAMPLE_RATE
      : pitchData.length > 0
        ? pitchData.length / PITCH_SAMPLE_RATE
        : blobDurationSec(selectedTrack.wavData);
  const contentWidth = durationToWidth(durationSec, timeScale);

  const notesToRender = tempNotes ?? trackNotes(selectedTrack);
  const pitchPolylinePoints = buildPitchPolylinePoints(pitchData, timeScale);

  const clientToRollCoords = useCallback(
    (clientX: number, clientY: number) => {
      const el = pianoRollRef.current;
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      return {
        x: clientX - rect.left + scrollLeftRef.current,
        y: clientY - rect.top + scrollTopRef.current,
      };
    },
    [pianoRollRef, scrollLeftRef, scrollTopRef],
  );

  const updateDraggedNote = useCallback(
    (clientX: number, clientY: number) => {
      if (draggedNoteIndex === null || !tempNotes) return;
      const coords = clientToRollCoords(clientX, clientY);
      if (!coords) return;
      const note = tempNotes[draggedNoteIndex];
      const newStart = Math.max(0, (coords.x - noteGrabOffsetXRef.current) / timeScale);
      const updated = [...tempNotes];
      updated[draggedNoteIndex] = {
        ...note,
        start: newStart,
        frequency: yToHz(coords.y),
      };
      setTempNotes(updated);
    },
    [clientToRollCoords, draggedNoteIndex, tempNotes, timeScale],
  );

  const finishNoteDrag = useCallback(async () => {
    if (draggedNoteIndex === null || !tempNotes) return;
    const notes = tempNotes;
    setDraggedNoteIndex(null);
    setTempNotes(null);
    if (onNoteDrop) {
      await onNoteDrop(notes);
    }
  }, [draggedNoteIndex, onNoteDrop, tempNotes]);

  useEffect(() => {
    if (draggedNoteIndex === null || !enableNoteDrag) return;

    const onMove = (e: MouseEvent) => updateDraggedNote(e.clientX, e.clientY);
    const onUp = () => {
      void finishNoteDrag();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [draggedNoteIndex, enableNoteDrag, finishNoteDrag, updateDraggedNote]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const el = pianoRollRef.current;
      if (!el) return;
      const ys: number[] = [];
      pitchForScroll
        .filter(hz => Number.isFinite(hz) && hz > 0)
        .forEach(hz => ys.push(pitchHzToDisplayY(hz)));
      notesToRender.forEach(note => {
        const y = noteFrequencyToRectY(note.frequency);
        if (y != null) ys.push(y + NOTE_HEIGHT / 2);
      });
      if (!ys.length) return;
      const midY = (Math.min(...ys) + Math.max(...ys)) / 2;
      const viewH = el.clientHeight;
      if (viewH <= 0) return;
      const maxScroll = Math.max(0, PIANO_ROLL_HEIGHT - viewH);
      syncScrollTopTo(Math.max(0, Math.min(maxScroll, midY - viewH / 2)));
    });
    return () => cancelAnimationFrame(frame);
  }, [selectedTrack.id, pitchForScroll, notesToRender, syncScrollTopTo]);

  const commitPitch = (pitch: number[]) => {
    if (!selectedTrack.features) return;
    const updated: TrackData = {
      ...selectedTrack,
      features: { ...selectedTrack.features, pitch },
    };
    setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
    setSelectedTrack(updated);
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditing || draggedNoteIndex !== null || isBusy) return;
    if (!selectedTrack.features) return;
    const coords = clientToRollCoords(event.clientX, event.clientY);
    if (!coords) return;
    setIsDraggingPitch(true);
    const next = [...selectedTrack.features.pitch];
    const timeIndex = Math.floor((coords.x / timeScale) * PITCH_SAMPLE_RATE);
    if (timeIndex >= 0 && timeIndex < next.length) {
      next[timeIndex] = yToHzContinuous(coords.y);
      setTempPitch(next);
    }
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (draggedNoteIndex !== null) {
      updateDraggedNote(event.clientX, event.clientY);
      return;
    }

    if (!isDraggingPitch || !tempPitch) return;
    const coords = clientToRollCoords(event.clientX, event.clientY);
    if (!coords) return;
    const timeIndex = Math.floor((coords.x / timeScale) * PITCH_SAMPLE_RATE);
    if (timeIndex >= 0 && timeIndex < tempPitch.length) {
      const next = [...tempPitch];
      next[timeIndex] = yToHzContinuous(coords.y);
      setTempPitch(next);
    }
  };

  const handleMouseUp = () => {
    if (draggedNoteIndex !== null) return;

    if (isDraggingPitch && tempPitch) {
      commitPitch(tempPitch);
    }
    setIsDraggingPitch(false);
    setTempPitch(null);
  };

  const handleNoteMouseDown = (e: React.MouseEvent<SVGRectElement>, index: number) => {
    if (!enableNoteDrag || isBusy) return;
    e.stopPropagation();
    const coords = clientToRollCoords(e.clientX, e.clientY);
    if (!coords) return;
    const notes = [...trackNotes(selectedTrack)];
    noteGrabOffsetXRef.current = coords.x - notes[index].start * timeScale;
    setDraggedNoteIndex(index);
    setTempNotes(notes);
  };

  const pianoRollCursor = isBusy
    ? 'wait'
    : isEditing
      ? 'crosshair'
      : enableNoteDrag
        ? 'default'
        : 'default';

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
        onScroll={syncScrollLeft}
      />
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <Box
          ref={keysRef}
          sx={{
            width: PIANO_ROLL_KEY_WIDTH,
            bgcolor: '#222',
            borderRight: '1px solid #333',
            overflowY: 'auto',
            overflowX: 'hidden',
            flexShrink: 0,
          }}
          onScroll={handleKeysScroll}
        >
          <PianoRollKeys />
        </Box>
        <Box
          ref={pianoRollRef}
          sx={{
            flex: 1,
            bgcolor: '#181818',
            position: 'relative',
            overflowX: 'auto',
            overflowY: 'auto',
            cursor: pianoRollCursor,
            opacity: isBusy ? 0.6 : 1,
            pointerEvents: isBusy ? 'none' : 'auto',
          }}
          onScroll={handlePianoRollScroll}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <Box sx={{ width: contentWidth, height: PIANO_ROLL_HEIGHT, position: 'relative' }}>
            <PlaybackCursor
              currentTime={currentTime}
              durationSec={durationSec}
              contentWidth={contentWidth}
            />
            <Box sx={{ position: 'absolute', inset: 0, zIndex: 0 }}>
              {octaves.map((oct, oi) =>
                keys.map((key, ki) => (
                  <Box
                    key={`${key.note}${oct}`}
                    sx={{
                      position: 'absolute',
                      top: (oi * keys.length + ki) * NOTE_HEIGHT,
                      width: '100%',
                      height: NOTE_HEIGHT,
                      bgcolor: key.isBlack ? 'rgba(34,34,34,0.7)' : 'rgba(255,255,255,0.07)',
                      border: '1px solid #333',
                    }}
                  />
                )),
              )}
            </Box>
            {notesToRender.length > 0 && (
              <svg
                width={contentWidth}
                height={PIANO_ROLL_HEIGHT}
                viewBox={`0 0 ${contentWidth} ${PIANO_ROLL_HEIGHT}`}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  zIndex: 1,
                  pointerEvents: enableNoteDrag && !isBusy ? 'all' : 'none',
                }}
              >
                {notesToRender.map((note, index) => {
                  const y = noteFrequencyToRectY(note.frequency);
                  const width = note.duration * timeScale;
                  if (y == null || width <= 0 || !Number.isFinite(note.start)) return null;
                  return (
                    <rect
                      key={`${index}-${note.start}-${note.frequency}`}
                      x={Math.max(0, note.start * timeScale)}
                      y={y}
                      width={width}
                      height={NOTE_HEIGHT}
                      fill={
                        draggedNoteIndex === index
                          ? 'rgba(255,215,0,0.5)'
                          : 'rgba(255,215,0,0.3)'
                      }
                      stroke="rgba(255,215,0,0.7)"
                      strokeWidth={draggedNoteIndex === index ? 2 : 1}
                      cursor={
                        enableNoteDrag && !isBusy
                          ? draggedNoteIndex === index
                            ? 'grabbing'
                            : 'grab'
                          : 'default'
                      }
                      onMouseDown={e => handleNoteMouseDown(e, index)}
                    />
                  );
                })}
              </svg>
            )}
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 2,
              }}
            >
              {pitchPolylinePoints.length > 0 && (
                <polyline
                  points={pitchPolylinePoints}
                  fill="none"
                  stroke="#646cff"
                  strokeWidth={2}
                />
              )}
            </svg>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
