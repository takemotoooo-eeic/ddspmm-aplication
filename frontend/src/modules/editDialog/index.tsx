import { Box, Button, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { useState } from 'react';
import { TrackData } from '../../types/trackData';

interface EditDialogProps {
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
}

export const EditDialog = ({ selectedTrack, tracks, setTracks }: EditDialogProps) => {
  const [editMode, setEditMode] = useState<'loudness' | 'pitch'>('pitch');
  const [isEditing, setIsEditing] = useState(false);

  const handleEditModeChange = (
    event: React.MouseEvent<HTMLElement>,
    newMode: 'loudness' | 'pitch' | null,
  ) => {
    if (newMode !== null) {
      setEditMode(newMode);
    }
  };

  const handleRegenerate = () => {
    // TODO: 波形再生成のロジックを実装
    console.log('波形を再生成します');
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
              minWidth: '80px',
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
        </Box>
      </Box>
      {/* ここにピアノロール本体が入る予定 */}
    </Box>
  );
};
