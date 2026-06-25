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
import type { NoteOperation, NoteOperationPayload } from '../../api/backend';
import { INSTRUMENTS, TIME_SCALE } from '../../constants/editor';
import type { Feature, Note } from '../../orval/models/backend-api';
import { AppMode } from '../../types/appMode';
import { ORIGINAL_TRACK_NAME, TrackData, trackNotes } from '../../types/trackData';
import { blobDurationSec, signalLengthFromDuration } from '../../utils/audio';
import { LoudnessEditor } from '../editors/LoudnessEditor';
import { NotesPianoRoll } from '../editors/NotesPianoRoll';
import { PitchEditor } from '../editors/PitchEditor';

type EditTab = 'pitch' | 'loudness';

type EditorScrollState = {
  left?: number;
  pitchTop?: number;
};

type FeatureParamPatch = Partial<Pick<Feature, 'pitch' | 'loudness' | 'z_feature'>>;

const editorScrollPositions = new Map<string, EditorScrollState>();

interface EditPanelProps {
  appMode: AppMode;
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
  numDenoisingSteps: number;
  useDdim: boolean;
}

export const EditPanel = ({
  appMode,
  currentTime,
  selectedTrack,
  tracks,
  setTracks,
  setSelectedTrack,
  onTimeLineClick,
  numDenoisingSteps,
  useDdim,
}: EditPanelProps) => {
  const [editTab, setEditTab] = useState<EditTab>('pitch');
  const [isEditing, setIsEditing] = useState(false);
  const [height, setHeight] = useState(480);
  const [isResizing, setIsResizing] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const noteDropRequestIdRef = useRef(0);
  const tracksRef = useRef(tracks);
  const selectedTrackRef = useRef(selectedTrack);
  tracksRef.current = tracks;
  selectedTrackRef.current = selectedTrack;

  const isFluidsynth = appMode === 'fluidsynth';
  const showInstrumentSelect = appMode === 'diffusion_ddsp';

  const instrumentOptions = Array.from(
    new Set([
      ...INSTRUMENTS,
      ...tracks
        .map(t => t.instrument)
        .filter(inst => inst !== ORIGINAL_TRACK_NAME),
    ]),
  );

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

  const regenerateDiffusionTrack = async (notes: Note[]) => {
    if (!selectedTrack.features) return;
    const signalLength = getSignalLength();
    const params = await generateDiffusionParams({
      notes,
      instrument_name: selectedTrack.instrument,
      signal_length: signalLength,
      num_denoising_steps: numDenoisingSteps,
      use_ddim: useDdim,
    });
    const wav = await generateDdspAudio(params);
    updateTrackWav(selectedTrack.id, wav, {
      features: {
        ...selectedTrack.features,
        ...params,
        notes,
      },
      notes,
      signalLength,
    });
  };

  const handleNoteDrop = (
    notes: Note[],
    operation: NoteOperation,
    operationPayload: NoteOperationPayload,
  ) => {
    if (appMode !== 'diffusion_ddsp' || !selectedTrack.features) return;

    const trackId = selectedTrack.id;
    const prevFeatures = {
      pitch: selectedTrack.features.pitch,
      loudness: selectedTrack.features.loudness,
      z_feature: selectedTrack.features.z_feature,
    };
    const notesUpdated: TrackData = {
      ...selectedTrack,
      notes,
      features: { ...selectedTrack.features, notes },
    };
    setTracks(tracks.map(t => (t.id === trackId ? notesUpdated : t)));
    setSelectedTrack(notesUpdated);

    if (!notes.length) return;

    const signalLength = getSignalLength();
    const instrument = selectedTrack.instrument;
    const requestId = ++noteDropRequestIdRef.current;

    void (async () => {
      try {
        const params = await generateDiffusionParams({
          notes,
          instrument_name: instrument,
          signal_length: signalLength,
          num_denoising_steps: numDenoisingSteps,
          use_ddim: useDdim,
          note_operation: operation,
          ...operationPayload,
          prev_features: prevFeatures,
        });
        if (requestId !== noteDropRequestIdRef.current) return;

        const wav = await generateDdspAudio(params);
        if (requestId !== noteDropRequestIdRef.current) return;

        const newTracks = tracksRef.current.map(t =>
          t.id === trackId && t.features
            ? {
              ...t,
              wavData: wav,
              features: { ...t.features, ...params, notes },
              notes,
              signalLength,
            }
            : t,
        );
        setTracks(newTracks);
        const currentSelected = selectedTrackRef.current;
        if (currentSelected?.id === trackId && currentSelected.features) {
          setSelectedTrack({
            ...currentSelected,
            wavData: wav,
            features: { ...currentSelected.features, ...params, notes },
            notes,
            signalLength,
          });
        }
      } catch (error) {
        console.error('Note drop regenerate error:', error);
      }
    })();
  };

  const handleFeatureParamsChange = (patch: FeatureParamPatch) => {
    if (!selectedTrack.features) return;

    const trackId = selectedTrack.id;
    const signalLength = getSignalLength();
    const updatedFeatures = { ...selectedTrack.features, ...patch };
    const updatedTrack: TrackData = {
      ...selectedTrack,
      features: updatedFeatures,
      signalLength,
    };
    setTracks(tracks.map(t => (t.id === trackId ? updatedTrack : t)));
    setSelectedTrack(updatedTrack);

    const requestId = ++noteDropRequestIdRef.current;
    setIsRegenerating(true);
    void (async () => {
      try {
        const wav = await generateDdspAudio({
          pitch: updatedFeatures.pitch,
          loudness: updatedFeatures.loudness,
          z_feature: updatedFeatures.z_feature,
        });
        if (requestId !== noteDropRequestIdRef.current) return;

        const newTracks = tracksRef.current.map(t =>
          t.id === trackId && t.features
            ? {
              ...t,
              wavData: wav,
              features: updatedFeatures,
              signalLength,
            }
            : t,
        );
        setTracks(newTracks);
        const currentSelected = selectedTrackRef.current;
        if (currentSelected?.id === trackId && currentSelected.features) {
          setSelectedTrack({
            ...currentSelected,
            wavData: wav,
            features: updatedFeatures,
            signalLength,
          });
        }
      } catch (error) {
        console.error('Direct feature edit regenerate error:', error);
      } finally {
        if (requestId === noteDropRequestIdRef.current) {
          setIsRegenerating(false);
        }
      }
    })();
  };

  const handleRegenerate = async () => {
    noteDropRequestIdRef.current += 1;
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

      if (appMode === 'diffusion_ddsp') {
        const notes = trackNotes(selectedTrack);
        if (!notes.length) {
          console.error('Notes not found');
          return;
        }
        await regenerateDiffusionTrack(notes);
        return;
      }

      const wav = await generateDdspAudio({
        pitch: selectedTrack.features.pitch,
        loudness: selectedTrack.features.loudness,
        z_feature: selectedTrack.features.z_feature,
      });
      updateTrackWav(selectedTrack.id, wav, {
        features: selectedTrack.features,
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
      return;
    }
    if (!selectedTrack.features) return;
    const updated: TrackData = {
      ...selectedTrack,
      features: { ...selectedTrack.features, notes },
    };
    setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
    setSelectedTrack(updated);
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
  const scrollState = editorScrollPositions.get(selectedTrack.id);

  const updateEditorScrollState = (patch: EditorScrollState) => {
    editorScrollPositions.set(selectedTrack.id, {
      ...editorScrollPositions.get(selectedTrack.id),
      ...patch,
    });
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
                  const instrument = e.target.value as typeof INSTRUMENTS[number];
                  const updated = { ...selectedTrack, instrument };
                  setTracks(tracks.map(t => (t.id === selectedTrack.id ? updated : t)));
                  setSelectedTrack(updated);
                }}
                sx={{ color: '#fff', bgcolor: '#333', height: 32 }}
              >
                {instrumentOptions.map(inst => (
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
            enableNoteDrag={appMode === 'diffusion_ddsp' && !isEditing}
            onNoteDrop={appMode === 'diffusion_ddsp' ? handleNoteDrop : undefined}
            onPitchChange={pitch => handleFeatureParamsChange({ pitch })}
            isBusy={isRegenerating}
            initialScrollPosition={
              scrollState == null
                ? undefined
                : { left: scrollState.left ?? 0, top: scrollState.pitchTop ?? 0 }
            }
            onScrollPositionChange={position =>
              updateEditorScrollState({ left: position.left, pitchTop: position.top })
            }
          />
        )}
        {!isFluidsynth && editTab === 'loudness' && (
          <LoudnessEditor
            {...editorProps}
            isBusy={isRegenerating}
            initialScrollLeft={scrollState?.left}
            onLoudnessChange={loudness => handleFeatureParamsChange({ loudness })}
            onScrollLeftChange={scrollLeft =>
              updateEditorScrollState({ left: scrollLeft })
            }
          />
        )}
      </Box>
    </Box>
  );
};
