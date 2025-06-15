import CloseIcon from '@mui/icons-material/Close';
import { Box, Button, IconButton, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { useRef, useState } from 'react';
import { TrackData } from '../../types/trackData';
import { Timeline } from '../timeLine';
import { PianoRollKeys, keys, octaves } from './PianoRollKeys';

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
  const [scrollPosition, setScrollPosition] = useState(0);
  const timelineRef = useRef<HTMLDivElement>(null);
  const pianoRollRef = useRef<HTMLDivElement>(null);

  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const newScrollPosition = event.currentTarget.scrollLeft;
    setScrollPosition(newScrollPosition);

    // タイムラインとピアノロールのスクロールを同期
    if (event.currentTarget === timelineRef.current && pianoRollRef.current) {
      pianoRollRef.current.scrollLeft = newScrollPosition;
    } else if (event.currentTarget === pianoRollRef.current && timelineRef.current) {
      timelineRef.current.scrollLeft = newScrollPosition;
    }
  };

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
        <Box sx={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
          {/* タイムライン */}
          <Box sx={{ position: 'relative', height: 20, display: 'flex', width: '100%' }}>
            <Box
              sx={{
                width: 80,
                bgcolor: '#222',
                borderRight: '1px solid #333',
                height: '100%',
                flexShrink: 0,
              }}
            />
            <Box
              ref={timelineRef}
              sx={{
                flex: 1,
                position: 'relative',
                overflowX: 'auto',
                overflowY: 'hidden',
                '&::-webkit-scrollbar': {
                  display: 'none',
                },
                msOverflowStyle: 'none',
                scrollbarWidth: 'none',
              }}
              onScroll={handleScroll}
            >
              <Box sx={{
                width: tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 2000,
                height: '100%',
              }}
                onClick={onTimeLineClick}>
                <Box
                  sx={{
                    position: 'absolute',
                    left: tracks.length > 0 ? (currentTime / (tracks[0].wavData.size / (16000 * 2))) * Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 0,
                    top: 0,
                    height: '100%',
                    width: 2,
                    bgcolor: 'rgba(255, 255, 255, 0.3)',
                    zIndex: 2,
                    pointerEvents: 'none',
                  }}
                />
                <Timeline
                  duration={tracks.length > 0 ? tracks[0].wavData.size / (16000 * 2) : 10}
                  width={tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 2000}
                  height={20}
                />
              </Box>
            </Box>
          </Box>
          {/* ピアノロール全体（鍵盤＋本体） */}
          <Box sx={{ flex: 1, display: 'flex', position: 'relative', overflowY: 'auto', height: '100%' }}>
            {/* 左端：ピアノロール（鍵盤） */}
            <Box
              sx={{
                width: 80,
                bgcolor: '#222',
                borderRight: '1px solid #333',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'sticky',
                left: 0,
                zIndex: 1,
                height: octaves.length * keys.length * 30,
              }}
            >
              <PianoRollKeys />
            </Box>
            {/* 右側：ピアノロール本体 */}
            <Box
              ref={pianoRollRef}
              sx={{
                flex: 1,
                bgcolor: '#181818',
                position: 'relative',
                minWidth: 0,
                height: octaves.length * keys.length * 30,
                overflowX: 'auto',
                '&::-webkit-scrollbar': {
                  display: 'none',
                },
                msOverflowStyle: 'none',
                scrollbarWidth: 'none',
              }}
              onScroll={handleScroll}
            >
              <Box sx={{
                width: tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 2000,
                height: '100%',
                position: 'relative',
              }}
                onClick={onTimeLineClick}>
                {/* 再生位置カーソル（全体） */}
                <Box
                  sx={{
                    position: 'absolute',
                    left: tracks.length > 0 ? (currentTime / (tracks[0].wavData.size / (16000 * 2))) * Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 0,
                    top: 0,
                    height: '100%',
                    width: 2,
                    bgcolor: 'rgba(255, 255, 255, 0.3)',
                    zIndex: 2,
                    pointerEvents: 'none',
                  }}
                />
                {/* 背景：白鍵・黒鍵の濃淡 */}
                <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}>
                  {octaves.map((oct: number, octaveIdx: number) =>
                    keys.map((key: { note: string; isBlack: boolean }, keyIdx: number) => (
                      <Box
                        key={`${key.note}${oct}`}
                        sx={{
                          position: 'absolute',
                          top: (octaveIdx * keys.length + keyIdx) * 30,
                          left: 0,
                          width: '100%',
                          height: 30,
                          bgcolor: key.isBlack ? 'rgba(34,34,34,0.7)' : 'rgba(255,255,255,0.07)',
                          border: '1px solid #333',
                        }}
                      />
                    ))
                  )}
                </Box>
              </Box>
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
