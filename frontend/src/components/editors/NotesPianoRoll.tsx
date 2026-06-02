import { Box } from '@mui/material';
import { useState } from 'react';
import { usePianoRollScroll } from '../../hooks/usePianoRollScroll';
import { NOTE_HEIGHT, PIANO_ROLL_KEY_WIDTH, TIME_SCALE } from '../../constants/editor';
import { keys, octaves, PIANO_ROLL_HEIGHT } from '../../constants/pianoRoll';
import type { Note } from '../../orval/models/backend-api';
import { TrackData, trackNotes, trackDurationSec } from '../../types/trackData';
import { durationToWidth } from '../../utils/audio';
import { hzToY, midiToRectY, yToHz } from '../../utils/pianoRollCoords';
import { snapHzToSemitone } from '../../utils/pitch';
import { PianoRollKeys } from './PianoRollKeys';
import { EditorTimelineRow } from './shared/EditorTimelineRow';
import { PlaybackCursor } from './shared/PlaybackCursor';

interface NotesPianoRollProps {
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
  isEditing: boolean;
  timeScale?: number;
  showPitchLine?: boolean;
  onNotesChange?: (notes: Note[]) => void;
}

export const NotesPianoRoll = ({
  currentTime,
  selectedTrack,
  tracks,
  setTracks,
  setSelectedTrack,
  onTimeLineClick,
  isEditing,
  timeScale = TIME_SCALE,
  showPitchLine = false,
  onNotesChange,
}: NotesPianoRollProps) => {
  const [draggedNoteIndex, setDraggedNoteIndex] = useState<number | null>(null);
  const [tempNotes, setTempNotes] = useState<Note[] | null>(null);
  const {
    timelineRef,
    pianoRollRef,
    keysRef,
    scrollLeftRef,
    scrollTopRef,
    syncScrollLeft,
    handlePianoRollScroll,
    handleKeysScroll,
  } = usePianoRollScroll();

  const durationSec = tracks.length > 0 ? trackDurationSec(tracks[0]) : 10;
  const contentWidth = durationToWidth(durationSec, timeScale);

  const commitNotes = (notes: Note[]) => {
    if (onNotesChange) {
      onNotesChange(notes);
      return;
    }
    const updatedTrack: TrackData = selectedTrack.features
      ? {
          ...selectedTrack,
          features: { ...selectedTrack.features, notes },
        }
      : { ...selectedTrack, notes };

    const updatedTracks = tracks.map(t =>
      t.id === selectedTrack.id ? updatedTrack : t,
    );
    setTracks(updatedTracks);
    setSelectedTrack(updatedTrack);
  };

  const handleNoteMouseDown = (
    event: React.MouseEvent<SVGRectElement>,
    noteIndex: number,
  ) => {
    if (!isEditing) return;
    event.stopPropagation();
    setDraggedNoteIndex(noteIndex);
    setTempNotes([...trackNotes(selectedTrack)]);
  };

  const handleNoteMouseMove = (event: React.MouseEvent) => {
    if (draggedNoteIndex === null || !tempNotes || !pianoRollRef.current) return;

    const rect = pianoRollRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left + scrollLeftRef.current;
    const y = event.clientY - rect.top + scrollTopRef.current;
    const note = tempNotes[draggedNoteIndex];
    const newFrequency = yToHz(y);
    const newStart = Math.max(0, (x - (note.duration * timeScale) / 2) / timeScale);

    const updated = [...tempNotes];
    updated[draggedNoteIndex] = {
      ...note,
      start: newStart,
      frequency: newFrequency,
    };
    setTempNotes(updated);
  };

  const handleNoteMouseUp = () => {
    if (draggedNoteIndex === null || !tempNotes) return;
    commitNotes(tempNotes);
    setDraggedNoteIndex(null);
    setTempNotes(null);
  };

  const notesToRender = tempNotes ?? trackNotes(selectedTrack);

  const renderNotes = () => (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: isEditing ? 'all' : 'none',
        zIndex: 1,
      }}
      onMouseMove={e => {
        if (draggedNoteIndex !== null) handleNoteMouseMove(e);
      }}
      onMouseUp={handleNoteMouseUp}
      onMouseLeave={handleNoteMouseUp}
    >
      {notesToRender.map((note, index) => {
        const midi = 12 * Math.log2(note.frequency / 440) + 69;
        const y = midiToRectY(midi);
        const x = note.start * timeScale;
        const width = note.duration * timeScale;
        const isDragged = draggedNoteIndex === index;
        return (
          <rect
            key={index}
            x={x}
            y={y}
            width={width}
            height={NOTE_HEIGHT}
            fill={isDragged ? 'rgba(255, 215, 0, 0.5)' : 'rgba(255, 215, 0, 0.3)'}
            stroke={isDragged ? 'rgba(255, 215, 0, 0.9)' : 'rgba(255, 215, 0, 0.6)'}
            strokeWidth={isDragged ? 2 : 1}
            cursor={isEditing ? 'move' : 'default'}
            onMouseDown={e => handleNoteMouseDown(e, index)}
          />
        );
      })}
    </svg>
  );

  const renderPitchLine = () => {
    if (!showPitchLine || !selectedTrack.features?.pitch) return null;
    const pitchData = selectedTrack.features.pitch;
    const points: string[] = [];
    const sampleRate = 31.25;
    pitchData.forEach((hz, index) => {
      if (hz > 0) {
        const x = (index / sampleRate) * timeScale;
        const y = hzToY(snapHzToSemitone(hz));
        points.push(`${x},${y}`);
      }
    });
    return (
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
        <polyline points={points.join(' ')} fill="none" stroke="#646cff" strokeWidth={2} />
      </svg>
    );
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
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
            minWidth: 0,
          }}
          onScroll={handlePianoRollScroll}
        >
          <Box sx={{ width: contentWidth, height: PIANO_ROLL_HEIGHT, position: 'relative' }}>
            <PlaybackCursor
              currentTime={currentTime}
              durationSec={durationSec}
              contentWidth={contentWidth}
            />
            <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}>
              {octaves.map((oct, octaveIdx) =>
                keys.map((key, keyIdx) => (
                  <Box
                    key={`${key.note}${oct}`}
                    sx={{
                      position: 'absolute',
                      top: (octaveIdx * keys.length + keyIdx) * NOTE_HEIGHT,
                      left: 0,
                      width: '100%',
                      height: NOTE_HEIGHT,
                      bgcolor: key.isBlack ? 'rgba(34,34,34,0.7)' : 'rgba(255,255,255,0.07)',
                      border: '1px solid #333',
                    }}
                  />
                )),
              )}
            </Box>
            {renderNotes()}
            {renderPitchLine()}
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
