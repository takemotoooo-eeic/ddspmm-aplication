import CloseIcon from '@mui/icons-material/Close';
import { Box, Button, IconButton, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { useRef, useState } from 'react';
import { useGenerateAudioFromDdsp } from '../../orval/backend-api';
import { DDSPGenerateParams } from '../../orval/models/backend-api';
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

// Hzからノート番号への変換関数
const hzToNoteNumber = (hz: number): number => {
  return 12 * Math.log2(hz / 440) + 69;
};

// ノート番号からY座標への変換関数
const noteNumberToY = (noteNumber: number): number => {
  const totalKeys = octaves.length * keys.length;
  const noteHeight = 30; // 各ノートの高さ
  return totalKeys * noteHeight - (noteNumber - 21) * noteHeight + 15;
};

// Y座標からノート番号への変換関数
const yToNoteNumber = (y: number): number => {
  const totalKeys = octaves.length * keys.length;
  const noteHeight = 30;
  return Math.round(totalKeys - (y - 15) / noteHeight + 21);
};

// ノート番号からHzへの変換関数
const noteNumberToHz = (noteNumber: number): number => {
  return 440 * Math.pow(2, (noteNumber - 69) / 12);
};

export const EditDialog = ({ currentTime, selectedTrack, tracks, setTracks, setSelectedTrack, onTimeLineClick }: EditDialogProps) => {
  const [editMode, setEditMode] = useState<'loudness' | 'pitch'>('pitch');
  const [isEditing, setIsEditing] = useState(false);
  const [scrollPosition, setScrollPosition] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartPoint, setDragStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [dragPoints, setDragPoints] = useState<{ x: number; y: number }[]>([]);
  const [tempPitch, setTempPitch] = useState<number[] | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const pianoRollRef = useRef<HTMLDivElement>(null);

  const { trigger: generateAudioTrigger } = useGenerateAudioFromDdsp();

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

  // マウスイベントハンドラー
  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditing || !selectedTrack) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left + scrollPosition;
    const y = event.clientY - rect.top;

    setIsDragging(true);
    setDragStartPoint({ x, y });
    setDragPoints([{ x, y }]);
    setTempPitch([...selectedTrack.features.pitch]);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || !dragStartPoint || !selectedTrack || !isEditing || !tempPitch) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left + scrollPosition;
    const y = event.clientY - rect.top;

    setDragPoints(prev => [...prev, { x, y }]);

    const timeScale = 200;
    const sampleRate = 31.25;
    const timeIndex = Math.floor((x / timeScale) * sampleRate);
    const noteNumber = yToNoteNumber(y);
    const newHz = noteNumberToHz(noteNumber);

    if (timeIndex >= 0 && timeIndex < selectedTrack.features.pitch.length) {
      const newPitch = [...tempPitch];
      newPitch[timeIndex] = newHz;
      setTempPitch(newPitch);
    }
  };

  const handleMouseUp = () => {
    if (!isDragging || !selectedTrack || !isEditing || !tempPitch) return;

    const updatedTrack = {
      ...selectedTrack,
      features: {
        ...selectedTrack.features,
        pitch: tempPitch
      }
    };

    const updatedTracks = tracks.map(track =>
      track.id === selectedTrack.id ? updatedTrack : track
    );

    setTracks(updatedTracks);
    setSelectedTrack(updatedTrack);
    setIsDragging(false);
    setDragStartPoint(null);
    setDragPoints([]);
    setTempPitch(null);
  };

  // ピッチデータを描画する関数
  const renderPitchLine = () => {
    if (!selectedTrack) return null;

    const points: { x: number; y: number }[] = [];
    const timeScale = 200;
    const sampleRate = 31.25;
    const pitchData = tempPitch || selectedTrack.features.pitch;

    pitchData.forEach((hz, index) => {
      if (hz > 0) {
        const noteNumber = hzToNoteNumber(hz);
        const x = (index / sampleRate) * timeScale;
        const y = noteNumberToY(noteNumber);
        points.push({ x, y });
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
          zIndex: 1,
        }}
      >
        <polyline
          points={points.map(p => `${p.x},${p.y}`).join(' ')}
          fill="none"
          stroke="#646cff"
          strokeWidth="2"
        />
      </svg>
    );
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
                cursor: isEditing ? 'crosshair' : 'default',
              }}
              onScroll={handleScroll}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <Box sx={{
                width: tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 2000,
                height: '100%',
                position: 'relative',
              }}
              >
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
                {renderPitchLine()}
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
