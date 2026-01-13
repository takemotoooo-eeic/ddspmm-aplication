import CloseIcon from '@mui/icons-material/Close';
import { Box, Button, IconButton, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { useEffect, useRef, useState } from 'react';
import { useGenerateAudioFromDdsp, useGenerateParamsFromDiffusion } from '../../orval/backend-api';
import { DDSPGenerateParams, DiffusionGenerateParams } from '../../orval/models/backend-api';
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
  const { trigger: generateParamsFromDiffusionTrigger } = useGenerateParamsFromDiffusion();

  const handleEditModeChange = (
    event: React.MouseEvent<HTMLElement>,
    newMode: 'loudness' | 'pitch' | null,
  ) => {
    if (newMode !== null) {
      setEditMode(newMode);
    }
  };

  const handleRegenerate = async () => {
    // 現在の楽器のnote情報と楽器名から合成パラメータを生成（diffusion/generate）
    if (!selectedTrack.features.notes || selectedTrack.features.notes.length === 0) {
      console.error('Notes not found');
      return;
    }

    try {
      // signal_lengthを計算（pitchの長さから逆算）
      const blockSize = 512;
      const pitchLength = selectedTrack.features.pitch.length;
      const signalLength = pitchLength * blockSize;

      const diffusionParams: DiffusionGenerateParams = {
        notes: selectedTrack.features.notes,
        instrument_name: selectedTrack.name,
        signal_length: signalLength,
      };

      // diffusion/generateで合成パラメータを生成
      const generatedParams = await generateParamsFromDiffusionTrigger(diffusionParams);

      // 生成したパラメータでddsp/generateで波形を生成
      const audioBody: DDSPGenerateParams = {
        pitch: generatedParams.pitch,
        loudness: generatedParams.loudness,
        z_feature: generatedParams.z_feature,
      };
      const response = await generateAudioTrigger(audioBody);
      const wavBlob = new Blob([await response.arrayBuffer()], { type: 'audio/wav' });

      // 生成したパラメータをfeaturesに反映
      const newTracks = tracks.map(track =>
        track.id === selectedTrack.id
          ? {
            ...track,
            wavData: wavBlob,
            features: {
              ...track.features,
              pitch: generatedParams.pitch,
              loudness: generatedParams.loudness,
              z_feature: generatedParams.z_feature,
            }
          }
          : track
      );
      setTracks(newTracks);

      // selectedTrackも更新して表示を反映
      const updatedTrack = newTracks.find(track => track.id === selectedTrack.id);
      if (updatedTrack) {
        setSelectedTrack(updatedTrack);
      }
    } catch (error) {
      console.error('Error generating audio:', error);
    }
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
          size="medium"
          sx={{
            '& .MuiToggleButton-root': {
              color: '#fff',
              borderColor: '#333',
              height: '32px',
              minWidth: '120px',
              fontSize: '1.2rem',
              '& .MuiSvgIcon-root': {
                fontSize: '1.2rem',
              },
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
          {/* <Box sx={{ color: '#fff', fontSize: '12px', minWidth: '60px' }}>
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
          </Box> */}
          <Button
            variant="contained"
            onClick={() => setIsEditing(!isEditing)}
            size="medium"
            sx={{
              bgcolor: isEditing ? '#646cff' : '#333',
              '&:hover': {
                bgcolor: isEditing ? '#535bf2' : '#444',
              },
              height: '32px',
              width: '120px',
              fontSize: '1.2rem',
              '& .MuiSvgIcon-root': {
                fontSize: '1.2rem',
              },
            }}
          >
            Edit
          </Button>
          <Button
            variant="contained"
            onClick={handleRegenerate}
            size="medium"
            sx={{
              bgcolor: '#646cff',
              '&:hover': {
                bgcolor: '#646cff',
              },
              height: '32px',
              minWidth: '200px',
              fontSize: '1.2rem',
              '& .MuiSvgIcon-root': {
                fontSize: '1.2rem',
              },
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
          />
        )}
      </Box>
    </Box>
  );
};
