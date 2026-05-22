import { Box } from '@mui/material';
import { useRef, useState } from 'react';
import {
  NOTE_HEIGHT,
  PIANO_ROLL_KEY_WIDTH,
  PITCH_SAMPLE_RATE,
  TIME_SCALE,
} from '../../constants/editor';
import { keys, octaves, PIANO_ROLL_HEIGHT } from '../../constants/pianoRoll';
import type { Note } from '../../orval/models/backend-api';
import { TrackData, trackNotes } from '../../types/trackData';
import { blobDurationSec, durationToWidth } from '../../utils/audio';
import { hzToYContinuous, midiToY, yToHz, yToHzContinuous } from '../../utils/pianoRollCoords';
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
  enableNoteEditing?: boolean;
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
  enableNoteEditing = false,
}: PitchEditorProps) => {
  const [isDraggingPitch, setIsDraggingPitch] = useState(false);
  const [tempPitch, setTempPitch] = useState<number[] | null>(null);
  const [draggedNoteIndex, setDraggedNoteIndex] = useState<number | null>(null);
  const [tempNotes, setTempNotes] = useState<Note[] | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const pianoRollRef = useRef<HTMLDivElement>(null);
  const scrollLeftRef = useRef(0);

  const durationSec = tracks.length > 0 ? blobDurationSec(tracks[0].wavData) : 10;
  const contentWidth = durationToWidth(durationSec, timeScale);

  const syncScroll = (scrollLeft: number) => {
    scrollLeftRef.current = scrollLeft;
    if (timelineRef.current) timelineRef.current.scrollLeft = scrollLeft;
    if (pianoRollRef.current) pianoRollRef.current.scrollLeft = scrollLeft;
  };

  const pitchData = tempPitch ?? selectedTrack.features?.pitch ?? [];
  const notesToRender = tempNotes ?? trackNotes(selectedTrack);

  const commitPitch = (pitch: number[]) => {
    if (!selectedTrack.features) return;
    const updated: TrackData = {
      ...selectedTrack,
      features: { ...selectedTrack.features, pitch },
    };
    setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
    setSelectedTrack(updated);
  };

  const commitNotes = (notes: Note[]) => {
    if (!selectedTrack.features) return;
    const updated: TrackData = {
      ...selectedTrack,
      features: { ...selectedTrack.features, notes },
    };
    setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
    setSelectedTrack(updated);
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditing || draggedNoteIndex !== null) return;
    if (!selectedTrack.features) return;
    const rect = pianoRollRef.current!.getBoundingClientRect();
    const x = event.clientX - rect.left + scrollLeftRef.current;
    const y = event.clientY - rect.top;
    setIsDraggingPitch(true);
    const next = [...selectedTrack.features.pitch];
    const timeIndex = Math.floor((x / timeScale) * PITCH_SAMPLE_RATE);
    if (timeIndex >= 0 && timeIndex < next.length) {
      next[timeIndex] = yToHzContinuous(y);
      setTempPitch(next);
    }
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (draggedNoteIndex !== null && tempNotes && pianoRollRef.current) {
      const rect = pianoRollRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left + scrollLeftRef.current;
      const y = event.clientY - rect.top;
      const note = tempNotes[draggedNoteIndex];
      const updated = [...tempNotes];
      updated[draggedNoteIndex] = {
        ...note,
        start: Math.max(0, (x - (note.duration * timeScale) / 2) / timeScale),
        frequency: yToHz(y),
      };
      setTempNotes(updated);
      return;
    }

    if (!isDraggingPitch || !tempPitch) return;
    const rect = pianoRollRef.current!.getBoundingClientRect();
    const x = event.clientX - rect.left + scrollLeftRef.current;
    const y = event.clientY - rect.top;
    const timeIndex = Math.floor((x / timeScale) * PITCH_SAMPLE_RATE);
    if (timeIndex >= 0 && timeIndex < tempPitch.length) {
      const next = [...tempPitch];
      next[timeIndex] = yToHzContinuous(y);
      setTempPitch(next);
    }
  };

  const handleMouseUp = () => {
    if (draggedNoteIndex !== null && tempNotes) {
      commitNotes(tempNotes);
      setDraggedNoteIndex(null);
      setTempNotes(null);
      return;
    }
    if (isDraggingPitch && tempPitch) {
      commitPitch(tempPitch);
    }
    setIsDraggingPitch(false);
    setTempPitch(null);
  };

  const handleNoteMouseDown = (e: React.MouseEvent<SVGRectElement>, index: number) => {
    if (!enableNoteEditing || !isEditing) return;
    e.stopPropagation();
    setDraggedNoteIndex(index);
    setTempNotes([...trackNotes(selectedTrack)]);
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
        onScroll={syncScroll}
      />
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <Box
          sx={{
            width: PIANO_ROLL_KEY_WIDTH,
            bgcolor: '#222',
            borderRight: '1px solid #333',
            overflowY: 'auto',
            flexShrink: 0,
          }}
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
            cursor: isEditing ? 'crosshair' : 'default',
          }}
          onScroll={e => syncScroll(e.currentTarget.scrollLeft)}
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
            {enableNoteEditing && notesToRender.length > 0 && (
              <svg
                style={{
                  position: 'absolute',
                  inset: 0,
                  zIndex: 1,
                  pointerEvents: isEditing ? 'all' : 'none',
                }}
              >
                {notesToRender.map((note, index) => {
                  const midi = 12 * Math.log2(note.frequency / 440) + 69;
                  const y = midiToY(midi) - NOTE_HEIGHT / 2;
                  return (
                    <rect
                      key={index}
                      x={note.start * timeScale}
                      y={y}
                      width={note.duration * timeScale}
                      height={NOTE_HEIGHT}
                      fill={
                        draggedNoteIndex === index
                          ? 'rgba(255,215,0,0.5)'
                          : 'rgba(255,215,0,0.3)'
                      }
                      stroke="rgba(255,215,0,0.7)"
                      cursor={isEditing ? 'move' : 'default'}
                      onMouseDown={e => handleNoteMouseDown(e, index)}
                    />
                  );
                })}
              </svg>
            )}
            <svg style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 2 }}>
              <polyline
                points={pitchData
                  .map((hz, i) =>
                    hz > 0 ? `${(i / PITCH_SAMPLE_RATE) * timeScale},${hzToYContinuous(hz)}` : null,
                  )
                  .filter(Boolean)
                  .join(' ')}
                fill="none"
                stroke="#646cff"
                strokeWidth={2}
              />
            </svg>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
