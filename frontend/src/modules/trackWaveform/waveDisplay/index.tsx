import { TrackData } from '@/types/trackData';
import { Box } from '@mui/material';
import { useEffect, useRef } from 'react';

interface WaveformDisplayProps {
  wavData: Blob | null;
  height?: number;
  width?: number;
  trackColor?: string;
  showTrackDivider?: boolean;
  track: TrackData;
  setSelectedTrack: (track: TrackData) => void;
}

export const WaveformDisplay = ({
  wavData,
  height = 60,
  width = 200,
  trackColor = '#646cff',
  showTrackDivider = true,
  track,
  setSelectedTrack,
}: WaveformDisplayProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!wavData || !canvasRef.current) return;

    const drawWaveform = async () => {
      const audioContext = new AudioContext();
      const arrayBuffer = await wavData.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const channelData = audioBuffer.getChannelData(0);
      const step = Math.ceil(channelData.length / width);
      const amp = height / 2;

      // 背景をクリア
      ctx.clearRect(0, 0, width, height);

      // トラック区切り線を描画
      if (showTrackDivider) {
        ctx.beginPath();
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        // 角丸の枠を描画（borderRadius: 8px相当）
        const radius = 8;
        ctx.moveTo(0.5 + radius, 0.5);
        ctx.lineTo(width - 0.5 - radius, 0.5);
        ctx.arcTo(width - 0.5, 0.5, width - 0.5, 0.5 + radius, radius);
        ctx.lineTo(width - 0.5, height - 0.5 - radius);
        ctx.arcTo(width - 0.5, height - 0.5, width - 0.5 - radius, height - 0.5, radius);
        ctx.lineTo(0.5 + radius, height - 0.5);
        ctx.arcTo(0.5, height - 0.5, 0.5, height - 0.5 - radius, radius);
        ctx.lineTo(0.5, 0.5 + radius);
        ctx.arcTo(0.5, 0.5, 0.5 + radius, 0.5, radius);
        ctx.stroke();
      }

      // 波形を描画
      ctx.beginPath();
      ctx.strokeStyle = '#000000'; // 波形を黒色に
      ctx.lineWidth = 2;

      for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;
        for (let j = 0; j < step; j++) {
          const datum = channelData[i * step + j];
          if (datum < min) min = datum;
          if (datum > max) max = datum;
        }
        ctx.moveTo(i, (1 + min) * amp);
        ctx.lineTo(i, (1 + max) * amp);
      }
      ctx.stroke();

      // 音データの色付きブロックを描画
      ctx.fillStyle = trackColor;
      ctx.globalAlpha = 0.3;
      ctx.fillRect(0, 0, width, height);
      ctx.globalAlpha = 1.0;

    };

    drawWaveform();
  }, [wavData, height, width, trackColor, showTrackDivider]);

  return (
    <Box
      sx={{
        width,
        height,
        bgcolor: '#1e1e1e',
        borderRadius: 1,
        overflow: 'hidden',
        borderBottom: '1px solid #333',
      }}
      onClick={() => setSelectedTrack(track)}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: '100%', height: '100%' }}
      />
    </Box>
  );
}; 
