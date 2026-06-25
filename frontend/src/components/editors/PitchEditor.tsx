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
import {
  type PianoRollScrollPosition,
  usePianoRollScroll,
} from '../../hooks/usePianoRollScroll';
import type { Note } from '../../orval/models/backend-api';
import { TrackData, trackNotes } from '../../types/trackData';
import { blobDurationSec, durationToWidth } from '../../utils/audio';
import {
  applyMonophonicInsert,
  applyMonophonicMove,
  buildNoteFromDrag,
  MIN_NOTE_DURATION_SEC,
  notesEqual,
  removeNote,
} from '../../utils/noteOverlap';
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

type NoteDragMode = 'move' | 'resize-start' | 'resize-end';

const NOTE_RESIZE_HANDLE_WIDTH = 6;

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
  /** ノートドロップ後に diffusion/generate → feature 更新（非同期） */
  onNoteDrop?: (notes: Note[], affectedRangeSec?: { start: number; end: number }) => void;
  isBusy?: boolean;
  initialScrollPosition?: PianoRollScrollPosition;
  onScrollPositionChange?: (position: PianoRollScrollPosition) => void;
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
  initialScrollPosition,
  onScrollPositionChange,
}: PitchEditorProps) => {
  const [isDraggingPitch, setIsDraggingPitch] = useState(false);
  const [tempPitch, setTempPitch] = useState<number[] | null>(null);
  const [draggedNoteIndex, setDraggedNoteIndex] = useState<number | null>(null);
  const [tempNotes, setTempNotes] = useState<Note[] | null>(null);
  const [draggedNotePreview, setDraggedNotePreview] = useState<Note | null>(null);
  const [noteDraw, setNoteDraw] = useState<{
    anchorTime: number;
    currentTime: number;
    frequency: number;
  } | null>(null);
  const [drawBaseNotes, setDrawBaseNotes] = useState<Note[] | null>(null);
  const noteDragOriginalRef = useRef<Note | null>(null);
  const noteDragBaseNotesRef = useRef<Note[] | null>(null);
  const noteDragModeRef = useRef<NoteDragMode>('move');
  const noteGrabOffsetXRef = useRef(0);
  const noteDragMovedRef = useRef(false);
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
  } = usePianoRollScroll(onScrollPositionChange);

  const pitchForScroll = selectedTrack.features?.pitch ?? [];
  const pitchData = tempPitch ?? pitchForScroll;
  const durationSec =
    selectedTrack.signalLength != null
      ? selectedTrack.signalLength / SAMPLE_RATE
      : pitchData.length > 0
        ? pitchData.length / PITCH_SAMPLE_RATE
        : blobDurationSec(selectedTrack.wavData);
  const contentWidth = durationToWidth(durationSec, timeScale);

  const notesToRender = (() => {
    if (noteDraw && drawBaseNotes) {
      const { start, duration, frequency } = buildNoteFromDrag(
        noteDraw.anchorTime,
        noteDraw.currentTime,
        noteDraw.frequency,
      );
      if (duration > 0) {
        return applyMonophonicInsert(drawBaseNotes, { start, duration, frequency });
      }
      return drawBaseNotes;
    }
    return tempNotes ?? trackNotes(selectedTrack);
  })();

  const drawingPreview =
    noteDraw &&
    (() => {
      const { start, duration, frequency } = buildNoteFromDrag(
        noteDraw.anchorTime,
        noteDraw.currentTime,
        noteDraw.frequency,
      );
      return { start, duration, frequency };
    })();

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
      const baseNotes = noteDragBaseNotesRef.current;
      const originalNote = noteDragOriginalRef.current;
      if (draggedNoteIndex === null || !baseNotes || !originalNote) return;
      const coords = clientToRollCoords(clientX, clientY);
      if (!coords) return;
      const mode = noteDragModeRef.current;
      const cursorTime = Math.max(0, coords.x / timeScale);
      const originalEnd = originalNote.start + originalNote.duration;
      const movedNote =
        mode === 'resize-start'
          ? {
              ...originalNote,
              start: Math.max(
                0,
                Math.min(cursorTime, originalEnd - MIN_NOTE_DURATION_SEC),
              ),
              duration:
                originalEnd -
                Math.max(0, Math.min(cursorTime, originalEnd - MIN_NOTE_DURATION_SEC)),
            }
          : mode === 'resize-end'
            ? {
                ...originalNote,
                duration: Math.max(MIN_NOTE_DURATION_SEC, cursorTime - originalNote.start),
              }
            : {
                ...originalNote,
                start: Math.max(0, (coords.x - noteGrabOffsetXRef.current) / timeScale),
                frequency: yToHz(coords.y),
              };
      if (
        movedNote.start !== originalNote.start ||
        movedNote.duration !== originalNote.duration ||
        movedNote.frequency !== originalNote.frequency
      ) {
        noteDragMovedRef.current = true;
      }
      const updated = applyMonophonicMove(baseNotes, draggedNoteIndex, movedNote);
      setDraggedNotePreview(movedNote);
      setTempNotes(updated);
    },
    [clientToRollCoords, draggedNoteIndex, timeScale],
  );

  const finishNoteDrag = useCallback(() => {
    if (draggedNoteIndex === null || !tempNotes) return;
    const notes = tempNotes;
    if (noteDragMovedRef.current) {
      const draggedNote = draggedNotePreview;
      const originalNote = noteDragOriginalRef.current;
      if (draggedNote && originalNote) {
        const start = Math.min(originalNote.start, draggedNote.start);
        const end = Math.max(
          originalNote.start + originalNote.duration,
          draggedNote.start + draggedNote.duration,
        );
        onNoteDrop?.(notes, { start, end });
      } else {
        onNoteDrop?.(notes);
      }
    }
    setDraggedNoteIndex(null);
    setTempNotes(null);
    setDraggedNotePreview(null);
    noteDragMovedRef.current = false;
    noteDragOriginalRef.current = null;
    noteDragBaseNotesRef.current = null;
    noteDragModeRef.current = 'move';
  }, [draggedNoteIndex, draggedNotePreview, onNoteDrop, tempNotes]);

  const finishNoteDraw = useCallback(() => {
    if (!noteDraw || !drawBaseNotes) return;
    const { start, duration, frequency } = buildNoteFromDrag(
      noteDraw.anchorTime,
      noteDraw.currentTime,
      noteDraw.frequency,
    );
    setNoteDraw(null);
    setDrawBaseNotes(null);
    if (duration < MIN_NOTE_DURATION_SEC) return;
    const finalNotes = applyMonophonicInsert(drawBaseNotes, { start, duration, frequency });
    onNoteDrop?.(finalNotes, { start, end: start + duration });
  }, [drawBaseNotes, noteDraw, onNoteDrop]);

  const updateNoteDraw = useCallback(
    (clientX: number) => {
      const coords = clientToRollCoords(clientX, 0);
      if (!coords) return;
      const currentTime = Math.max(0, coords.x / timeScale);
      setNoteDraw(prev => (prev ? { ...prev, currentTime } : null));
    },
    [clientToRollCoords, timeScale],
  );

  useEffect(() => {
    if (!noteDraw || !enableNoteDrag) return;

    const onMove = (e: MouseEvent) => updateNoteDraw(e.clientX);
    const onUp = () => finishNoteDraw();

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [noteDraw, enableNoteDrag, finishNoteDraw, updateNoteDraw]);

  useEffect(() => {
    if (draggedNoteIndex === null || !enableNoteDrag) return;

    const onMove = (e: MouseEvent) => updateDraggedNote(e.clientX, e.clientY);
    const onUp = () => {
      finishNoteDrag();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [draggedNoteIndex, enableNoteDrag, finishNoteDrag, updateDraggedNote]);

  useEffect(() => {
    if (!initialScrollPosition) return;
    const frame = requestAnimationFrame(() => {
      syncScrollLeft(initialScrollPosition.left);
      syncScrollTopTo(initialScrollPosition.top);
    });
    return () => cancelAnimationFrame(frame);
  }, [
    initialScrollPosition,
    selectedTrack.id,
    syncScrollLeft,
    syncScrollTopTo,
  ]);

  useEffect(() => {
    if (initialScrollPosition) return;
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
  }, [initialScrollPosition, selectedTrack.id, pitchForScroll, notesToRender, syncScrollTopTo]);

  const commitPitch = (pitch: number[]) => {
    if (!selectedTrack.features) return;
    const updated: TrackData = {
      ...selectedTrack,
      features: { ...selectedTrack.features, pitch },
    };
    setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
    setSelectedTrack(updated);
  };

  const handlePianoRollBackgroundMouseDown = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!enableNoteDrag || isEditing || isBusy || noteDraw) return;
    if (event.target !== event.currentTarget) return;

    const coords = clientToRollCoords(event.clientX, event.clientY);
    if (!coords) return;

    const anchorTime = Math.max(0, coords.x / timeScale);
    const frequency = yToHz(coords.y);
    setDrawBaseNotes([...trackNotes(selectedTrack)]);
    setNoteDraw({ anchorTime, currentTime: anchorTime, frequency });
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditing || draggedNoteIndex !== null || isBusy || noteDraw) return;
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
    if (draggedNoteIndex !== null || noteDraw) return;

    if (isDraggingPitch && tempPitch) {
      commitPitch(tempPitch);
    }
    setIsDraggingPitch(false);
    setTempPitch(null);
  };

  const handleNoteMouseDown = (
    e: React.MouseEvent<SVGRectElement>,
    index: number,
    mode: NoteDragMode = 'move',
  ) => {
    if (!enableNoteDrag || isBusy || noteDraw) return;
    e.stopPropagation();
    const coords = clientToRollCoords(e.clientX, e.clientY);
    if (!coords) return;
    const notes = [...trackNotes(selectedTrack)];
    noteGrabOffsetXRef.current = coords.x - notes[index].start * timeScale;
    noteDragMovedRef.current = false;
    noteDragOriginalRef.current = { ...notes[index] };
    noteDragBaseNotesRef.current = notes;
    noteDragModeRef.current = mode;
    setDraggedNotePreview(notes[index]);
    setDraggedNoteIndex(index);
    setTempNotes(notes);
  };

  const handleNoteDoubleClick = (e: React.MouseEvent<SVGRectElement>, note: Note) => {
    if (!enableNoteDrag || isBusy || noteDraw) return;
    e.stopPropagation();
    e.preventDefault();
    setDraggedNoteIndex(null);
    setTempNotes(null);
    setDraggedNotePreview(null);
    noteDragMovedRef.current = false;
    noteDragOriginalRef.current = null;
    noteDragBaseNotesRef.current = null;
    noteDragModeRef.current = 'move';
    const notes = removeNote(trackNotes(selectedTrack), note);
    onNoteDrop?.(notes, { start: note.start, end: note.start + note.duration });
  };

  const pianoRollCursor = isBusy ? 'wait' : isEditing || enableNoteDrag ? 'crosshair' : 'default';

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
            {enableNoteDrag && (
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
                onMouseDown={handlePianoRollBackgroundMouseDown}
              >
                {notesToRender.map((note, index) => {
                  const y = noteFrequencyToRectY(note.frequency);
                  const width = note.duration * timeScale;
                  if (y == null || width <= 0 || !Number.isFinite(note.start)) return null;
                  const isDragged =
                    draggedNotePreview != null && notesEqual(note, draggedNotePreview);
                  const isNewDrawn =
                    drawingPreview &&
                    note.start === drawingPreview.start &&
                    note.duration === drawingPreview.duration &&
                    note.frequency === drawingPreview.frequency;
                  const x = Math.max(0, note.start * timeScale);
                  const handleWidth = Math.min(NOTE_RESIZE_HANDLE_WIDTH, width / 2);
                  return (
                    <g key={`${index}-${note.start}-${note.frequency}-${note.duration}`}>
                      <rect
                        x={x}
                        y={y}
                        width={width}
                        height={NOTE_HEIGHT}
                        fill={
                          isNewDrawn
                            ? 'rgba(100,255,150,0.45)'
                            : isDragged
                              ? 'rgba(255,215,0,0.5)'
                              : 'rgba(255,215,0,0.3)'
                        }
                        stroke={
                          isNewDrawn ? 'rgba(100,255,150,0.9)' : 'rgba(255,215,0,0.7)'
                        }
                        strokeWidth={isDragged || isNewDrawn ? 2 : 1}
                        cursor={
                          enableNoteDrag && !isBusy && !noteDraw
                            ? isDragged
                              ? 'grabbing'
                              : 'grab'
                            : 'default'
                        }
                        onMouseDown={e => handleNoteMouseDown(e, index)}
                        onDoubleClick={e => handleNoteDoubleClick(e, note)}
                      >
                        <title>Drag to move, drag edges to resize, double-click to delete</title>
                      </rect>
                      {handleWidth > 0 && (
                        <>
                          <rect
                            x={x}
                            y={y}
                            width={handleWidth}
                            height={NOTE_HEIGHT}
                            fill="transparent"
                            cursor="ew-resize"
                            onMouseDown={e => handleNoteMouseDown(e, index, 'resize-start')}
                          />
                          <rect
                            x={x + width - handleWidth}
                            y={y}
                            width={handleWidth}
                            height={NOTE_HEIGHT}
                            fill="transparent"
                            cursor="ew-resize"
                            onMouseDown={e => handleNoteMouseDown(e, index, 'resize-end')}
                          />
                        </>
                      )}
                    </g>
                  );
                })}
                {drawingPreview && drawingPreview.duration <= 0 && (
                  (() => {
                    const y = noteFrequencyToRectY(drawingPreview.frequency);
                    if (y == null) return null;
                    return (
                      <rect
                        x={drawingPreview.start * timeScale}
                        y={y}
                        width={2}
                        height={NOTE_HEIGHT}
                        fill="rgba(100,255,150,0.45)"
                        stroke="rgba(100,255,150,0.9)"
                        strokeWidth={2}
                        pointerEvents="none"
                      />
                    );
                  })()
                )}
              </svg>
            )}
            {!enableNoteDrag && notesToRender.length > 0 && (
              <svg
                width={contentWidth}
                height={PIANO_ROLL_HEIGHT}
                viewBox={`0 0 ${contentWidth} ${PIANO_ROLL_HEIGHT}`}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  zIndex: 1,
                  pointerEvents: 'none',
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
                      fill="rgba(255,215,0,0.3)"
                      stroke="rgba(255,215,0,0.7)"
                      strokeWidth={1}
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
                  strokeWidth={3}
                />
              )}
            </svg>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
