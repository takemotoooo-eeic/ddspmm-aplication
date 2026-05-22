import CloseIcon from '@mui/icons-material/Close';
import {
  Box,
  Button,
  CircularProgress,
  FormControl,
  IconButton,
  MenuItem,
  Select,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import { useEffect, useRef, useState } from 'react';
import {
  generateDdspAudio,
  generateDiffusionParams,
  generateFluidsynthAudio,
} from '../../api/backend';
import { INSTRUMENTS, Instrument, TIME_SCALE } from '../../constants/editor';
import { LoudnessEditor } from '../editors/LoudnessEditor';
import { NotesPianoRoll } from '../editors/NotesPianoRoll';
import { PitchEditor } from '../editors/PitchEditor';
import type { Note } from '../../orval/models/backend-api';
import { AppMode } from '../../types/appMode';
import { TrackData, trackNotes } from '../../types/trackData';
import { blobDurationSec, signalLengthFromDuration } from '../../utils/audio';

type EditTab = 'pitch' | 'loudness';

interface EditPanelProps {
  appMode: AppMode;
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
}

export const EditPanel = ({
  appMode,
  currentTime,
  selectedTrack,
  tracks,
  setTracks,
  setSelectedTrack,
  onTimeLineClick,
}: EditPanelProps) => {
  const [editTab, setEditTab] = useState<EditTab>('pitch');
  const [isEditing, setIsEditing] = useState(false);
  const [height, setHeight] = useState(480);
  const [isResizing, setIsResizing] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  const isFluidsynth = appMode === 'fluidsynth';
  const showInstrumentSelect = appMode === 'diffusion_ddsp';

  const getSignalLength = (): number => {
    if (selectedTrack.signalLength) return selectedTrack.signalLength;
    if (selectedTrack.features?.pitch.length) {
      return selectedTrack.features.pitch.length * 512;
    }
    return signalLengthFromDuration(blobDurationSec(selectedTrack.wavData));
  };

  const updateTrackWav = (trackId: string, wavData: Blob, patch: Partial<TrackData>) => {
    const newTracks = tracks.map(t =>
      t.id === trackId ? { ...t, wavData, ...patch } : t,
    );
    setTracks(newTracks);
    const updated = newTracks.find(t => t.id === trackId);
    if (updated) setSelectedTrack(updated);
  };

  const handleRegenerate = async () => {
    setIsRegenerating(true);
    try {
      const signalLength = getSignalLength();

      if (appMode === 'fluidsynth') {
        const wav = await generateFluidsynthAudio({
          notes: trackNotes(selectedTrack),
          instrument_name: selectedTrack.instrument,
          signal_length: signalLength,
        });
        updateTrackWav(selectedTrack.id, wav, {
          notes: trackNotes(selectedTrack),
          signalLength,
        });
        return;
      }

      if (!selectedTrack.features) return;

      let params = {
        pitch: selectedTrack.features.pitch,
        loudness: selectedTrack.features.loudness,
        z_feature: selectedTrack.features.z_feature,
      };

      if (appMode === 'diffusion_ddsp') {
        const notes = trackNotes(selectedTrack);
        if (!notes.length) {
          console.error('Notes not found');
          return;
        }
        params = await generateDiffusionParams({
          notes,
          instrument_name: selectedTrack.instrument,
          signal_length: signalLength,
        });
      }

      const wav = await generateDdspAudio(params);
      updateTrackWav(selectedTrack.id, wav, {
        features: { ...selectedTrack.features, ...params },
        signalLength,
      });
    } catch (error) {
      console.error('Regenerate error:', error);
    } finally {
      setIsRegenerating(false);
    }
  };

  const handleNotesChange = (notes: Note[]) => {
    if (isFluidsynth) {
      const updated: TrackData = { ...selectedTrack, notes };
      setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
      setSelectedTrack(updated);
    }
  };

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newHeight = window.innerHeight - e.clientY;
      if (newHeight >= 200 && newHeight <= window.innerHeight - 140) {
        setHeight(newHeight);
      }
    };
    const onUp = () => setIsResizing(false);
    if (isResizing) {
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    }
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [isResizing]);

  const editorProps = {
    currentTime,
    selectedTrack,
    tracks,
    setTracks,
    setSelectedTrack,
    onTimeLineClick,
    isEditing,
    timeScale: TIME_SCALE,
  };

  return (
    <Box
      ref={panelRef}
      sx={{
        position: 'fixed',
        left: 0,
        bottom: 0,
        width: '100vw',
        height: `${height}px`,
        bgcolor: '#181818',
        borderTop: '1px solid #333',
        zIndex: 20,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box
        sx={{
          height: 8,
          bgcolor: '#333',
          cursor: 'ns-resize',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        onMouseDown={e => {
          e.preventDefault();
          setIsResizing(true);
        }}
      >
        <Box sx={{ width: 40, height: 4, bgcolor: '#666', borderRadius: 1 }} />
      </Box>

      <Box
        sx={{
          height: 48,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #333',
          px: 2,
        }}
      >
        {!isFluidsynth ? (
          <ToggleButtonGroup
            value={editTab}
            exclusive
            onChange={(_, v: EditTab | null) => v && setEditTab(v)}
            size="small"
          >
            <ToggleButton value="pitch">Pitch</ToggleButton>
            <ToggleButton value="loudness">Loudness</ToggleButton>
          </ToggleButtonGroup>
        ) : (
          <Box sx={{ color: '#fff', fontSize: '1.1rem' }}>Notes</Box>
        )}

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          {showInstrumentSelect && (
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={selectedTrack.instrument}
                onChange={e => {
                  const instrument = e.target.value as Instrument;
                  const updated = { ...selectedTrack, instrument };
                  setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
                  setSelectedTrack(updated);
                }}
                sx={{ color: '#fff', bgcolor: '#333', height: 32 }}
              >
                {INSTRUMENTS.map(inst => (
                  <MenuItem key={inst} value={inst}>
                    {inst.toUpperCase()}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          <Button
            variant="contained"
            onClick={() => setIsEditing(!isEditing)}
            sx={{ bgcolor: isEditing ? '#646cff' : '#333', height: 32, minWidth: 80 }}
          >
            Edit
          </Button>
          <Button
            variant="contained"
            onClick={handleRegenerate}
            disabled={isRegenerating}
            sx={{ bgcolor: '#646cff', height: 32, minWidth: 160 }}
          >
            {isRegenerating ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={16} sx={{ color: '#fff' }} />
                REGENERATING...
              </Box>
            ) : (
              'REGENERATE'
            )}
          </Button>
          <IconButton
            onClick={() => {
              setIsEditing(false);
              setSelectedTrack(null);
            }}
            sx={{ color: '#fff' }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      </Box>

      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        {isFluidsynth && (
          <NotesPianoRoll {...editorProps} onNotesChange={handleNotesChange} />
        )}
        {!isFluidsynth && editTab === 'pitch' && (
          <PitchEditor
            {...editorProps}
            enableNoteEditing={appMode === 'diffusion_ddsp'}
          />
        )}
        {!isFluidsynth && editTab === 'loudness' && (
          <LoudnessEditor {...editorProps} />
        )}
      </Box>
    </Box>
  );
};
