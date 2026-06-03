import { Box, Menu, MenuItem } from '@mui/material';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  TTM_EDGE_SCROLL_MARGIN_PX,
  TTM_EDGE_SCROLL_SPEED_PX,
} from '../../../constants/editor';
import { TrackData } from '../../../types/trackData';
import { downloadWavBlob, safeAudioFilename } from '../../../utils/audio';
import { normalizeSelection } from '../../../utils/waveformSelection';
import {
  applyEdgeScroll,
  clampTtmSelectionPx,
  shouldEdgeScroll,
  waveformXFromClient,
} from '../../../utils/ttmWaveformDrag';

interface WaveformDisplayProps {
  wavData: Blob | null;
  height?: number;
  width?: number;
  trackColor?: string;
  backgroundColor?: string;
  showTrackDivider?: boolean;
  selectable?: boolean;
  regionSelectEnabled?: boolean;
  durationSec?: number;
  scrollContainerRef?: React.RefObject<HTMLElement | null>;
  onRegionSelected?: (
    startSec: number,
    endSec: number,
    anchor: { top: number; left: number },
  ) => void;
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
  regionSelectEnabled = false,
  durationSec = 0,
  scrollContainerRef,
  onRegionSelected,
  track,
  setSelectedTrack,
}: WaveformDisplayProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragStartPx = useRef<number | null>(null);
  const lastClientX = useRef<number | null>(null);
  const [menuPosition, setMenuPosition] = useState<{ top: number; left: number } | null>(
    null,
  );
  const [selectionPx, setSelectionPx] = useState<{ start: number; end: number } | null>(
    null,
  );
  const [isDragging, setIsDragging] = useState(false);

  const edgeScrollEnabled =
    regionSelectEnabled && scrollContainerRef != null;

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

  const resolveSelectionPx = useCallback(
    (clientX: number): { start: number; end: number } | null => {
      if (dragStartPx.current == null || !containerRef.current) return null;
      let endPx = waveformXFromClient(clientX, containerRef.current, width);
      let startPx = dragStartPx.current;
      if (edgeScrollEnabled) {
        const clamped = clampTtmSelectionPx(startPx, endPx, width, durationSec);
        endPx = clamped.endPx;
      }
      return { start: startPx, end: endPx };
    },
    [width, durationSec, edgeScrollEnabled],
  );

  const updateDragEnd = useCallback(
    (clientX: number) => {
      const sel = resolveSelectionPx(clientX);
      if (sel) setSelectionPx(sel);
    },
    [resolveSelectionPx],
  );

  const finishSelection = useCallback(
    (clientX: number, clientY: number) => {
      if (dragStartPx.current == null || !onRegionSelected || !containerRef.current) {
        return;
      }
      const sel = resolveSelectionPx(clientX);
      if (!sel) return;
      const { startSec, endSec } = normalizeSelection(
        sel.start,
        sel.end,
        width,
        durationSec,
      );
      dragStartPx.current = null;
      lastClientX.current = null;
      setIsDragging(false);
      setSelectionPx(null);
      onRegionSelected(startSec, endSec, { top: clientY, left: clientX });
    },
    [onRegionSelected, width, durationSec, resolveSelectionPx],
  );

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!regionSelectEnabled || !onRegionSelected || durationSec <= 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const x = waveformXFromClient(event.clientX, event.currentTarget, width);
    dragStartPx.current = x;
    lastClientX.current = event.clientX;
    setSelectionPx({ start: x, end: x });
    setIsDragging(true);
  };

  useEffect(() => {
    if (!isDragging) return;

    const onPointerMove = (event: PointerEvent) => {
      lastClientX.current = event.clientX;
      updateDragEnd(event.clientX);
    };

    const onPointerUp = (event: PointerEvent) => {
      finishSelection(event.clientX, event.clientY);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
    };
  }, [isDragging, updateDragEnd, finishSelection]);

  useEffect(() => {
    if (!isDragging || !edgeScrollEnabled) return;

    let rafId = 0;
    const tick = () => {
      const scroller = scrollContainerRef?.current;
      const clientX = lastClientX.current;
      if (scroller && clientX != null && containerRef.current && dragStartPx.current != null) {
        const endPx = waveformXFromClient(clientX, containerRef.current, width);
        if (
          shouldEdgeScroll(
            dragStartPx.current,
            endPx,
            clientX,
            scroller,
            TTM_EDGE_SCROLL_MARGIN_PX,
            width,
            durationSec,
          )
        ) {
          const scrolled = applyEdgeScroll(
            scroller,
            clientX,
            TTM_EDGE_SCROLL_MARGIN_PX,
            TTM_EDGE_SCROLL_SPEED_PX,
          );
          if (scrolled) {
            updateDragEnd(clientX);
          }
        } else {
          updateDragEnd(clientX);
        }
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isDragging, edgeScrollEnabled, scrollContainerRef, updateDragEnd, width, durationSec]);

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
      drawPlaceholder('No audio data');
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
        if (!cancelled) drawPlaceholder('Unable to display waveform');
      } finally {
        await audioContext.close();
      }
    };

    drawWaveform();

    return () => {
      cancelled = true;
    };
  }, [wavData, height, width, trackColor, backgroundColor, showTrackDivider]);

  const selLeft = selectionPx ? Math.min(selectionPx.start, selectionPx.end) : 0;
  const selWidth = selectionPx
    ? Math.abs(selectionPx.end - selectionPx.start)
    : 0;

  return (
    <>
      <Box
        ref={containerRef}
        sx={{
          width,
          height,
          bgcolor: backgroundColor,
          borderRadius: 1,
          overflow: 'hidden',
          borderBottom: '1px solid #333',
          cursor: regionSelectEnabled ? 'crosshair' : selectable ? 'pointer' : 'default',
          position: 'relative',
          userSelect: 'none',
          touchAction: 'none',
        }}
        onClick={() => {
          if (!regionSelectEnabled && selectable) setSelectedTrack(track);
        }}
        onContextMenu={handleContextMenu}
        onPointerDown={handlePointerDown}
      >
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ width: '100%', height: '100%', pointerEvents: 'none' }}
        />
        {selectionPx && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: selLeft,
              width: Math.max(selWidth, 2),
              height: '100%',
              bgcolor: 'rgba(100, 108, 255, 0.35)',
              border: '1px solid #646cff',
              pointerEvents: 'none',
            }}
          />
        )}
      </Box>
      <Menu
        open={menuPosition != null}
        onClose={() => setMenuPosition(null)}
        anchorReference="anchorPosition"
        anchorPosition={
          menuPosition ? { top: menuPosition.top, left: menuPosition.left } : undefined
        }
      >
        <MenuItem onClick={handleDownloadWav}>Download WAV</MenuItem>
      </Menu>
    </>
  );
};
