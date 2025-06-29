import { Box } from '@mui/material';
import { useRef, useState } from 'react';
import { TrackData } from '../../../types/trackData';
import { Timeline } from '../../timeLine';
import { PianoRollKeys, keys, octaves } from './pianoRollKeys';

interface PitchEditorProps {
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

// Hzからノート番号への変換関数
const hzToNoteNumber = (hz: number): number => {
  return 12 * Math.log2(hz / 440) + 67;
};

// ノート番号からY座標への変換関数
const noteNumberToY = (noteNumber: number, verticalZoomLevel: number = 1): number => {
  const totalKeys = octaves.length * keys.length;
  const baseNoteHeight = 30; // 各ノートの基本高さ
  const noteHeight = baseNoteHeight * verticalZoomLevel; // ズームレベルに応じて高さを調整
  return totalKeys * noteHeight - (noteNumber - 21) * noteHeight + 15;
};

// Y座標からノート番号への変換関数
const yToNoteNumber = (y: number, verticalZoomLevel: number = 1): number => {
  const totalKeys = octaves.length * keys.length;
  const baseNoteHeight = 30;
  const noteHeight = baseNoteHeight * verticalZoomLevel;
  return Math.round(totalKeys - (y - 15) / noteHeight + 21);
};

// ノート番号からHzへの変換関数
const noteNumberToHz = (noteNumber: number): number => {
  return 440 * Math.pow(2, (noteNumber - 67) / 12);
};

export const PitchEditor = ({
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
}: PitchEditorProps) => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartPoint, setDragStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [dragPoints, setDragPoints] = useState<{ x: number; y: number }[]>([]);
  const [tempPitch, setTempPitch] = useState<number[] | null>(null);
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

    const sampleRate = 31.25;
    const timeIndex = Math.floor((x / timeScale) * sampleRate);
    const noteNumber = yToNoteNumber(y, verticalZoomLevel);
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
    regenerate();
  };

  // ピッチデータを描画する関数
  const renderPitchLine = () => {
    if (!selectedTrack) return null;

    const points: { x: number; y: number }[] = [];
    const sampleRate = 31.25;
    const pitchData = tempPitch || selectedTrack.features.pitch;

    pitchData.forEach((hz: number, index: number) => {
      if (hz > 0) {
        const noteNumber = hzToNoteNumber(hz);
        const x = (index / sampleRate) * timeScale;
        const y = noteNumberToY(noteNumber, verticalZoomLevel);
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
            height: octaves.length * keys.length * 30 * verticalZoomLevel,
          }}
        >
          <PianoRollKeys verticalZoomLevel={verticalZoomLevel} />
        </Box>
        {/* 右側：ピアノロール本体 */}
        <Box
          ref={pianoRollRef}
          sx={{
            flex: 1,
            bgcolor: '#181818',
            position: 'relative',
            minWidth: 0,
            height: octaves.length * keys.length * 30 * verticalZoomLevel,
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
            width: tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * timeScale) : 2000,
            height: '100%',
            position: 'relative',
          }}
          >
            {/* 再生位置カーソル（全体） */}
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
            {/* 背景：白鍵・黒鍵の濃淡 */}
            <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}>
              {octaves.map((oct: number, octaveIdx: number) =>
                keys.map((key: { note: string; isBlack: boolean }, keyIdx: number) => (
                  <Box
                    key={`${key.note}${oct}`}
                    sx={{
                      position: 'absolute',
                      top: (octaveIdx * keys.length + keyIdx) * 30 * verticalZoomLevel,
                      left: 0,
                      width: '100%',
                      height: 30 * verticalZoomLevel,
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
  );
};
