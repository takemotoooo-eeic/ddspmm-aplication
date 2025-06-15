import CloseIcon from '@mui/icons-material/Close';
import { Box, Button, IconButton, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { useState } from 'react';
import { TrackData } from '../../types/trackData';
import { Timeline } from '../timeLine';

interface EditDialogProps {
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
}

export const EditDialog = ({ selectedTrack, tracks, setTracks, setSelectedTrack }: EditDialogProps) => {
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
            }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      </Box>
      {editMode === 'pitch' && (
        <Box sx={{ display: 'flex', height: '392px' }}>
          {/* 左端：ピアノロール（鍵盤） */}
          <Box sx={{ width: '60px', bgcolor: '#222', display: 'flex', flexDirection: 'column', alignItems: 'center', pt: 2 }}>
            <Box sx={{ height: '30px', bgcolor: '#1e1e1e', borderBottom: '1px solid #333' }} />
            {/* C3〜C5のラベルを縦に並べる */}
            {['C5', 'C4', 'C3'].map((note) => (
              <Box key={note} sx={{ height: '120px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', borderBottom: '1px solid #333', width: '100%' }}>
                {note}
              </Box>
            ))}
          </Box>
          {/* 右側：タイムライン＋ピアノロール本体 */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {/* タイムライン */}
            <Timeline
              duration={tracks.length > 0 ? tracks[0].wavData.size / (16000 * 2) : 10}
              width={tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 2000}
              height={20}
            />
            {/* ピアノロール本体 */}
            <Box sx={{ flex: 1, bgcolor: '#181818' }}>
              {/* ここにノートや波形を描画予定 */}
            </Box>
          </Box>
        </Box>
      )}
      {editMode === 'loudness' && (
        <Box>
          <Typography>Loudness</Typography>
        </Box>
      )}
    </Box>
  );
};
