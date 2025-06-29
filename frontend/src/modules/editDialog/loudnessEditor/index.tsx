import { Box } from '@mui/material';
import { useRef, useState } from 'react';
import { TrackData } from '../../../types/trackData';
import { Timeline } from '../../timeLine';

interface LoudnessEditorProps {
  currentTime: number;
  selectedTrack: TrackData;
  tracks: TrackData[];
  setTracks: (tracks: TrackData[]) => void;
  setSelectedTrack: (track: TrackData | null) => void;
  onTimeLineClick: (event: React.MouseEvent<HTMLDivElement>) => void;
  isEditing: boolean;
  timeScale: number;
  verticalZoomLevel: number;
  regenerate: () => void;
}

// dBからY座標への変換関数
const dbToY = (db: number, verticalZoomLevel: number = 1): number => {
  const minDb = -80;
  const maxDb = 0;
  const baseHeight = 600;
  const height = baseHeight * verticalZoomLevel;
  return height - ((db - minDb) / (maxDb - minDb)) * height;
};

// Y座標からdBへの変換関数
const yToDb = (y: number, verticalZoomLevel: number = 1): number => {
  const minDb = -80;
  const maxDb = 0;
  const baseHeight = 600;
  const height = baseHeight * verticalZoomLevel;
  return minDb + ((height - y) / height) * (maxDb - minDb);
};

export const LoudnessEditor = ({
  currentTime,
  selectedTrack,
  tracks,
  setTracks,
  setSelectedTrack,
  onTimeLineClick,
  isEditing,
  timeScale,
  verticalZoomLevel,
  regenerate
}: LoudnessEditorProps) => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartPoint, setDragStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [dragPoints, setDragPoints] = useState<{ x: number; y: number }[]>([]);
  const [tempLoudness, setTempLoudness] = useState<number[] | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const newScrollPosition = event.currentTarget.scrollLeft;
    setScrollPosition(newScrollPosition);

    // タイムラインとエディターのスクロールを同期
    if (event.currentTarget === timelineRef.current && editorRef.current) {
      editorRef.current.scrollLeft = newScrollPosition;
    } else if (event.currentTarget === editorRef.current && timelineRef.current) {
      timelineRef.current.scrollLeft = newScrollPosition;
    }
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
    setTempLoudness([...selectedTrack.features.loudness]);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || !dragStartPoint || !selectedTrack || !isEditing || !tempLoudness) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left + scrollPosition;
    const y = event.clientY - rect.top;

    setDragPoints(prev => [...prev, { x, y }]);

    const sampleRate = 31.25;
    const timeIndex = Math.floor((x / timeScale) * sampleRate);
    const newDb = yToDb(y, verticalZoomLevel);

    if (timeIndex >= 0 && timeIndex < selectedTrack.features.loudness.length) {
      const newLoudness = [...tempLoudness];
      newLoudness[timeIndex] = newDb;
      setTempLoudness(newLoudness);
    }
  };

  const handleMouseUp = () => {
    if (!isDragging || !selectedTrack || !isEditing || !tempLoudness) return;

    const updatedTrack = {
      ...selectedTrack,
      features: {
        ...selectedTrack.features,
        loudness: tempLoudness
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
    setTempLoudness(null);
    regenerate();
  };

  // ラウドネスデータを描画する関数
  const renderLoudnessLine = () => {
    if (!selectedTrack) return null;

    const points: { x: number; y: number }[] = [];
    const sampleRate = 31.25;
    const loudnessData = tempLoudness || selectedTrack.features.loudness;

    loudnessData.forEach((db: number, index: number) => {
      const x = (index / sampleRate) * timeScale;
      const y = dbToY(db, verticalZoomLevel);
      points.push({ x, y });
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

  // dBスケールのメモリを描画する関数
  const renderDbScale = () => {
    const dbValues = [0, -10, -20, -30, -40, -50, -60, -70, -80];
    return (
      <Box
        sx={{
          width: 80,
          bgcolor: '#222',
          borderRight: '1px solid #333',
          position: 'sticky',
          left: 0,
          zIndex: 1,
          height: 600 * verticalZoomLevel,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          py: 1,
        }}
      >
        {dbValues.map((db) => (
          <Box
            key={db}
            sx={{
              color: '#fff',
              fontSize: '12px',
              textAlign: 'right',
              pr: 1,
              borderBottom: '1px solid #333',
              pb: 0.5,
            }}
          >
            {db}
          </Box>
        ))}
      </Box>
    );
  };

  return (
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
            position: 'sticky',
            borderBottom: '1px solid #333',
            left: 0,
            zIndex: 2,
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
            width: tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * timeScale) : 2000,
            height: '100%',
          }}
            onClick={onTimeLineClick}>
            <Box
              sx={{
                position: 'absolute',
                left: tracks.length > 0 ? (currentTime / (tracks[0].wavData.size / (16000 * 2))) * Math.floor((tracks[0].wavData.size / (16000 * 2)) * timeScale) : 0,
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
              width={tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * timeScale) : 2000}
              height={20}
            />
          </Box>
        </Box>
      </Box>
      {/* エディター全体 */}
      <Box sx={{ flex: 1, display: 'flex', position: 'relative', overflowY: 'auto', height: '100%' }}>
        {/* 左端：dBスケール */}
        {renderDbScale()}
        {/* 右側：エディター本体 */}
        <Box
          ref={editorRef}
          sx={{
            flex: 1,
            bgcolor: '#181818',
            position: 'relative',
            minWidth: 0,
            height: 600 * verticalZoomLevel,
            overflowX: 'auto',
            overflowY: 'auto',
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
            width: tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * timeScale) : 2000,
            height: '100%',
            position: 'relative',
            minHeight: '100%',
          }}
          >
            {/* 再生位置カーソル */}
            <Box
              sx={{
                position: 'absolute',
                left: tracks.length > 0 ? (currentTime / (tracks[0].wavData.size / (16000 * 2))) * Math.floor((tracks[0].wavData.size / (16000 * 2)) * timeScale) : 0,
                top: 0,
                height: '100%',
                width: 2,
                bgcolor: 'rgba(255, 255, 255, 0.3)',
                zIndex: 2,
                pointerEvents: 'none',
              }}
            />
            {renderLoudnessLine()}
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
