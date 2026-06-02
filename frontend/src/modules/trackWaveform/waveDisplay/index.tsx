import { TrackData } from '../../../types/trackData';
import { downloadWavBlob, safeAudioFilename } from '../../../utils/audio';
import { Box, Menu, MenuItem } from '@mui/material';
import { useEffect, useRef, useState } from 'react';

interface WaveformDisplayProps {
  wavData: Blob | null;
  height?: number;
  width?: number;
  trackColor?: string;
  backgroundColor?: string;
  showTrackDivider?: boolean;
  selectable?: boolean;
  track: TrackData;
  setSelectedTrack: (track: TrackData) => void;
}

export const WaveformDisplay = ({
  wavData,
  height = 60,
  width = 200,
  trackColor = '#646cff',
  backgroundColor = '#1e1e1e',
  showTrackDivider = true,
  selectable = true,
  track,
  setSelectedTrack,
}: WaveformDisplayProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [menuPosition, setMenuPosition] = useState<{ top: number; left: number } | null>(
    null,
  );

  const handleContextMenu = (event: React.MouseEvent) => {
    event.preventDefault();
    if (!wavData || wavData.size === 0) return;
    setMenuPosition({ top: event.clientY, left: event.clientX });
  };

  const handleDownloadWav = () => {
    if (!wavData || wavData.size === 0) return;
    downloadWavBlob(wavData, `${safeAudioFilename(track.name)}.wav`);
    setMenuPosition(null);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawPlaceholder = (message: string) => {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#666';
      ctx.font = '12px sans-serif';
      ctx.fillText(message, 8, height / 2);
    };

    if (!wavData || wavData.size === 0) {
      drawPlaceholder('音声データなし');
      return;
    }

    let cancelled = false;

    const drawWaveform = async () => {
      const audioContext = new AudioContext();
      try {
        const arrayBuffer = await wavData.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));

        if (cancelled) return;

        const channelData = audioBuffer.getChannelData(0);
        const step = Math.max(1, Math.ceil(channelData.length / width));
        const amp = height / 2;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = backgroundColor;
        ctx.fillRect(0, 0, width, height);

        if (showTrackDivider) {
          ctx.beginPath();
          ctx.strokeStyle = '#e0e0e0';
          ctx.lineWidth = 1;
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

        ctx.beginPath();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;

        for (let i = 0; i < width; i++) {
          let min = 1.0;
          let max = -1.0;
          for (let j = 0; j < step; j++) {
            const idx = i * step + j;
            if (idx >= channelData.length) break;
            const datum = channelData[idx] * 5;
            if (datum < min) min = datum;
            if (datum > max) max = datum;
          }
          ctx.moveTo(i, (1 + min) * amp);
          ctx.lineTo(i, (1 + max) * amp);
        }
        ctx.stroke();

        ctx.fillStyle = trackColor;
        ctx.globalAlpha = 0.3;
        ctx.fillRect(0, 0, width, height);
        ctx.globalAlpha = 1.0;
      } catch (error) {
        console.error('Failed to decode WAV for waveform display:', error);
        if (!cancelled) drawPlaceholder('波形を表示できません');
      } finally {
        await audioContext.close();
      }
    };

    drawWaveform();

    return () => {
      cancelled = true;
    };
  }, [wavData, height, width, trackColor, backgroundColor, showTrackDivider]);

  return (
    <>
      <Box
        sx={{
          width,
          height,
          bgcolor: backgroundColor,
          borderRadius: 1,
          overflow: 'hidden',
          borderBottom: '1px solid #333',
          cursor: selectable ? 'pointer' : 'default',
        }}
        onClick={() => {
          if (selectable) setSelectedTrack(track);
        }}
        onContextMenu={handleContextMenu}
      >
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ width: '100%', height: '100%' }}
        />
      </Box>
      <Menu
        open={menuPosition != null}
        onClose={() => setMenuPosition(null)}
        anchorReference="anchorPosition"
        anchorPosition={
          menuPosition ? { top: menuPosition.top, left: menuPosition.left } : undefined
        }
      >
        <MenuItem onClick={handleDownloadWav}>WAVをダウンロード</MenuItem>
      </Menu>
    </>
  );
}; 
