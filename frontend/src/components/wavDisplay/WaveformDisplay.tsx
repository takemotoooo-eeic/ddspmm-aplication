import { Box } from '@mui/material';
import { useEffect, useRef } from 'react';

interface WaveformDisplayProps {
  wavData: Blob | null;
  height?: number;
  width?: number;
}

export const WaveformDisplay = ({ wavData, height = 60, width = 200 }: WaveformDisplayProps) => {
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

      ctx.clearRect(0, 0, width, height);
      ctx.beginPath();
      ctx.strokeStyle = '#646cff';
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
    };

    drawWaveform();
  }, [wavData, height, width]);

  return (
    <Box sx={{ width, height, bgcolor: '#1e1e1e', borderRadius: 1 }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: '100%', height: '100%' }}
      />
    </Box>
  );
}; 
