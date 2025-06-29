import CloseIcon from '@mui/icons-material/Close';
import { Box, Button, IconButton, Slider, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { useEffect, useRef, useState } from 'react';
import { useGenerateAudioFromDdsp } from '../../orval/backend-api';
import { DDSPGenerateParams } from '../../orval/models/backend-api';
import { TrackData } from '../../types/trackData';
import { LoudnessEditor } from './loudnessEditor';
import { PitchEditor } from './pitchEditor';

interface EditDialogProps {
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
  setZoomLevel: (zoomLevel: number) => void;
  zoomLevel: number;
  timeScale: number;
}

export const EditDialog = ({ currentTime, selectedTrack, tracks, setTracks, setSelectedTrack, onTimeLineClick, setZoomLevel, zoomLevel, timeScale }: EditDialogProps) => {
  const [editMode, setEditMode] = useState<'loudness' | 'pitch'>('pitch');
  const [isEditing, setIsEditing] = useState(false);
  const [height, setHeight] = useState(480);
  const [isResizing, setIsResizing] = useState(false);
  const [verticalZoomLevel, setVerticalZoomLevel] = useState(1);
  const dialogRef = useRef<HTMLDivElement>(null);

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

  // リサイズハンドラーのマウスダウンイベント
  const handleResizeMouseDown = (event: React.MouseEvent) => {
    event.preventDefault();
    setIsResizing(true);
  };

  // マウス移動とマウスアップのイベントリスナー
  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!isResizing) return;

      const newHeight = window.innerHeight - event.clientY;
      const minHeight = 200;
      const maxHeight = window.innerHeight - 140;

      if (newHeight >= minHeight && newHeight <= maxHeight) {
        setHeight(newHeight);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  return (
    <Box
      ref={dialogRef}
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
      {/* リサイズハンドル */}
      <Box
        sx={{
          width: '100%',
          height: '8px',
          bgcolor: '#333',
          cursor: 'ns-resize',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          '&:hover': {
            bgcolor: '#444',
          },
          '&:active': {
            bgcolor: '#555',
          },
        }}
        onMouseDown={handleResizeMouseDown}
      >
        <Box
          sx={{
            width: '40px',
            height: '4px',
            bgcolor: '#666',
            borderRadius: '2px',
          }}
        />
      </Box>

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
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mr: 2 }}>
          <Box sx={{ color: '#fff', fontSize: '12px', minWidth: '60px' }}>
            H Zoom: {zoomLevel.toFixed(1)}x
          </Box>
          <Box sx={{ minWidth: 80, maxWidth: 120, width: '100px' }}>
            <Slider
              value={zoomLevel}
              onChange={(_, value) => setZoomLevel(value as number)}
              min={0.5}
              max={5}
              step={0.1}
              sx={{
                color: '#646cff',
                '& .MuiSlider-thumb': {
                  bgcolor: '#646cff',
                  width: '16px',
                  height: '16px',
                  '&:hover': {
                    width: '20px',
                    height: '20px',
                  },
                },
                '& .MuiSlider-track': {
                  bgcolor: '#646cff',
                  height: '4px',
                },
                '& .MuiSlider-rail': {
                  height: '4px',
                },
              }}
            />
          </Box>
          <Box sx={{ color: '#fff', fontSize: '12px', minWidth: '60px' }}>
            V Zoom: {verticalZoomLevel.toFixed(1)}x
          </Box>
          <Box sx={{ minWidth: 80, maxWidth: 120, width: '100px' }}>
            <Slider
              value={verticalZoomLevel}
              onChange={(_, value) => setVerticalZoomLevel(value as number)}
              min={0.5}
              max={2}
              step={0.1}
              sx={{
                color: '#646cff',
                '& .MuiSlider-thumb': {
                  bgcolor: '#646cff',
                  width: '16px',
                  height: '16px',
                  '&:hover': {
                    width: '20px',
                    height: '20px',
                  },
                },
                '& .MuiSlider-track': {
                  bgcolor: '#646cff',
                  height: '4px',
                },
                '& .MuiSlider-rail': {
                  height: '4px',
                },
              }}
            />
          </Box>
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

      {/* エディターコンテンツ */}
      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        {editMode === 'pitch' && (
          <PitchEditor
            currentTime={currentTime}
            selectedTrack={selectedTrack}
            tracks={tracks}
            setTracks={setTracks}
            setSelectedTrack={setSelectedTrack}
            onTimeLineClick={onTimeLineClick}
            isEditing={isEditing}
            timeScale={timeScale}
            verticalZoomLevel={verticalZoomLevel}
            regenerate={handleRegenerate}
          />
        )}
        {editMode === 'loudness' && (
          <LoudnessEditor
            currentTime={currentTime}
            selectedTrack={selectedTrack}
            tracks={tracks}
            setTracks={setTracks}
            setSelectedTrack={setSelectedTrack}
            onTimeLineClick={onTimeLineClick}
            isEditing={isEditing}
            timeScale={timeScale}
            verticalZoomLevel={verticalZoomLevel}
            regenerate={handleRegenerate}
          />
        )}
      </Box>
    </Box>
  );
};
