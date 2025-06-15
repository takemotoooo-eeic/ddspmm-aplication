import CloseIcon from '@mui/icons-material/Close';
import { Box, Button, IconButton, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { useState } from 'react';
import { useGenerateAudioFromDdsp } from '../../orval/backend-api';
import { DDSPGenerateParams } from '../../orval/models/backend-api';
import { TrackData } from '../../types/trackData';
import { PitchEditor } from './pitchEditor';

interface EditDialogProps {
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
}

export const EditDialog = ({ currentTime, selectedTrack, tracks, setTracks, setSelectedTrack, onTimeLineClick }: EditDialogProps) => {
  const [editMode, setEditMode] = useState<'loudness' | 'pitch'>('pitch');
  const [isEditing, setIsEditing] = useState(false);

  const { trigger: generateAudioTrigger } = useGenerateAudioFromDdsp();

  const handleEditModeChange = (
    event: React.MouseEvent<HTMLElement>,
    newMode: 'loudness' | 'pitch' | null,
  ) => {
    if (newMode !== null) {
      setEditMode(newMode);
    }
  };

  const handleRegenerate = async () => {
    // TODO: 波形再生成のロジックを実装
    console.log('波形を再生成します');
    const body: DDSPGenerateParams = {
      z_feature: selectedTrack.features.z_feature,
      loudness: selectedTrack.features.loudness,
      pitch: selectedTrack.features.pitch,
    };
    const response = await generateAudioTrigger(body);
    const wavBlob = new Blob([await response.arrayBuffer()], { type: 'audio/wav' });
    const newTracks = tracks.map(track =>
      track.id === selectedTrack.id ? { ...track, wavData: wavBlob } : track
    );
    setTracks(newTracks);
  };

  return (
    <Box sx={{
      position: 'fixed',
      left: 0,
      bottom: 0,
      width: '100vw',
      height: '480px',
      bgcolor: '#181818',
      borderTop: '1px solid #333',
      zIndex: 20,
    }}>
      {/* ツールバー部分 */}
      <Box sx={{
        width: '100%',
        height: '48px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #333',
        px: 2,
      }}>
        <ToggleButtonGroup
          value={editMode}
          exclusive
          onChange={handleEditModeChange}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: '#fff',
              borderColor: '#333',
              height: '32px',
              minWidth: '80px',
              '&.Mui-selected': {
                bgcolor: '#333',
                color: '#fff',
              },
            },
          }}
        >
          <ToggleButton value="pitch">Pitch</ToggleButton>
          <ToggleButton value="loudness">Loudness</ToggleButton>
        </ToggleButtonGroup>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Button
            variant="contained"
            onClick={() => setIsEditing(!isEditing)}
            sx={{
              bgcolor: isEditing ? '#646cff' : '#333',
              '&:hover': {
                bgcolor: isEditing ? '#535bf2' : '#444',
              },
              height: '32px',
              width: '80px',
            }}
          >
            Edit
          </Button>
          <Button
            variant="contained"
            onClick={handleRegenerate}
            sx={{
              bgcolor: '#646cff',
              '&:hover': {
                bgcolor: '#646cff',
              },
              height: '32px',
              minWidth: '120px',
            }}
          >
            REGENERATE
          </Button>
          <IconButton
            onClick={() => {
              setIsEditing(false);
              setSelectedTrack(null);
            }}
            sx={{
              color: '#fff',
              '&:hover': {
                bgcolor: '#333',
              },
              height: '24px',
              width: '24px',
            }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      </Box>
      {editMode === 'pitch' && (
        <PitchEditor
          currentTime={currentTime}
          selectedTrack={selectedTrack}
          tracks={tracks}
          setTracks={setTracks}
          setSelectedTrack={setSelectedTrack}
          onTimeLineClick={onTimeLineClick}
          isEditing={isEditing}
        />
      )}
      {editMode === 'loudness' && (
        <Box>
          <Typography>Loudness</Typography>
        </Box>
      )}
    </Box>
  );
};
