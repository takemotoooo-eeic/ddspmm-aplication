import { useCallback, useEffect, useRef, useState } from 'react';
import { trackDurationSec } from '../types/trackData';
import type { TrackData } from '../types/trackData';

export function useAudioPlayback(tracks: TrackData[]) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const isPlayingRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioBuffersRef = useRef<Map<string, AudioBuffer>>(new Map());
  const sourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const playbackStartTimeRef = useRef(0);
  const animationFrameRef = useRef<number>(0);

  const trackDuration =
    tracks.length > 0 ? trackDurationSec(tracks[0]) : 0;

  const handleStop = useCallback(() => {
    isPlayingRef.current = false;
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    sourcesRef.current.forEach(source => {
      try {
        source.stop();
      } catch {
        /* already stopped */
      }
    });
    sourcesRef.current = [];
    audioBuffersRef.current.clear();
  }, []);

  const updateCurrentTime = useCallback(() => {
    if (!isPlayingRef.current || !audioContextRef.current) return;

    const newTime =
      audioContextRef.current.currentTime - playbackStartTimeRef.current;

    if (newTime >= trackDuration) {
      handleStop();
      return;
    }

    if (newTime >= 0) {
      setCurrentTime(newTime);
    }
    animationFrameRef.current = requestAnimationFrame(updateCurrentTime);
  }, [trackDuration, handleStop]);

  const handlePlay = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }

    if (isPlayingRef.current) return;

    const startTime = audioContextRef.current.currentTime;
    playbackStartTimeRef.current = startTime - currentTime;
    sourcesRef.current = [];

    try {
      for (const track of tracks) {
        if (track.muted) continue;

        if (!audioBuffersRef.current.has(track.id)) {
          const arrayBuffer = await track.wavData.arrayBuffer();
          const audioBuffer =
            await audioContextRef.current.decodeAudioData(arrayBuffer);
          audioBuffersRef.current.set(track.id, audioBuffer);
        }

        const source = audioContextRef.current.createBufferSource();
        const gainNode = audioContextRef.current.createGain();
        source.buffer = audioBuffersRef.current.get(track.id)!;
        gainNode.gain.value = track.volume;
        source.connect(gainNode);
        gainNode.connect(audioContextRef.current.destination);
        source.start(0, currentTime);
        sourcesRef.current.push(source);
      }

      isPlayingRef.current = true;
      setIsPlaying(true);
      updateCurrentTime();
    } catch (error) {
      console.error('Error during playback:', error);
      isPlayingRef.current = false;
      setIsPlaying(false);
    }
  }, [tracks, currentTime, updateCurrentTime]);

  const seekToTime = useCallback(
    (newTime: number) => {
      if (isPlayingRef.current) return;
      setCurrentTime(newTime);
      playbackStartTimeRef.current =
        (audioContextRef.current?.currentTime || 0) - newTime;
    },
    [],
  );

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      audioContextRef.current?.close();
    };
  }, []);

  return {
    isPlaying,
    currentTime,
    trackDuration,
    handlePlay,
    handleStop,
    seekToTime,
    isPlayingRef,
    playbackStartTimeRef,
    audioContextRef,
  };
}
