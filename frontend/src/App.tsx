import {
  AppBar,
  Box,
  createTheme,
  CssBaseline,
  ThemeProvider,
  Toolbar,
  Typography
} from '@mui/material';
import { useEffect, useRef, useState } from 'react';
import { AddButton } from './components/buttons/ImportButton';
import { RefreshButton } from './components/buttons/refreshButton';
import { StartButton } from './components/buttons/StartButton';
import { StopButton } from './components/buttons/StopButton';
import { useDisclosure } from './hooks/useDisclosure';
import { EditDialog } from './modules/editDialog';
import { ImportTrackDialog } from './modules/importTrackDialog';
import { Timeline } from './modules/timeLine';
import { TrackSidebar } from './modules/trackSidebar';
import { TrackRowWaveform } from './modules/trackWaveform';
import { useGenerateAudioFromDdsp, useTrainDdsp } from './orval/backend-api';
import { BodyTrainDdspDdspTrainPost, DDSPGenerateParams } from './orval/models/backend-api';
import { TrackData } from './types/trackData';


// ダークテーマ
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#646cff' },
    background: {
      default: '#242424',
      paper: '#1e1e1e'
    },
    text: { primary: '#ffffff' }
  }
});

export default function App() {
  const [wavFile, setWavFile] = useState<File | null>(null);
  const [midFile, setMidFile] = useState<File | null>(null);
  const [tracks, setTracks] = useState<TrackData[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<TrackData | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const isPlayingRef = useRef(false);
  const [currentTime, setCurrentTime] = useState(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioBuffersRef = useRef<Map<string, AudioBuffer>>(new Map());
  const startTimeRef = useRef<number>(0);
  const animationFrameRef = useRef<number>(0);
  const sourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const playbackStartTimeRef = useRef<number>(0);

  const handleTrackClick = (track: TrackData) => {
    setSelectedTrack(track);
  };

  const handleMuteToggle = (trackId: string) => {
    setTracks(prevTracks =>
      prevTracks.map(track =>
        track.id === trackId
          ? { ...track, muted: !track.muted }
          : track
      )
    );
  };

  const handleVolumeChange = (trackId: string, volume: number) => {
    setTracks(prevTracks =>
      prevTracks.map(track =>
        track.id === trackId
          ? { ...track, volume }
          : track
      )
    );
  };

  const { trigger: trainTrigger } = useTrainDdsp();
  const { trigger: generateAudioTrigger } = useGenerateAudioFromDdsp();

  const {
    isOpen: isOpenImportTracksDialog,
    open: openImportTracksDialog,
    close: closeImportTracksDialog,
  } = useDisclosure({});

  // 時間をフォーマットする関数
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // 現在時間の更新
  const updateCurrentTime = () => {
    if (!isPlayingRef.current || !audioContextRef.current) {
      return;
    }

    const newTime = audioContextRef.current.currentTime - playbackStartTimeRef.current;

    // トラックの終端に達したかチェック
    const trackDuration = tracks.length > 0 ? tracks[0].wavData.size / (16000 * 2) : 0;
    if (newTime >= trackDuration) {
      handleStop();
      return;
    }

    if (newTime >= 0) {
      setCurrentTime(newTime);
    }

    // 次のフレームで再度更新をスケジュール
    animationFrameRef.current = requestAnimationFrame(updateCurrentTime);
  };

  // 再生処理
  const handlePlay = async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }

    if (!isPlayingRef.current) {
      const startTime = audioContextRef.current.currentTime;
      startTimeRef.current = startTime;
      // 現在のカーソル位置を再生開始位置として設定
      playbackStartTimeRef.current = startTime - currentTime;
      sourcesRef.current = [];

      try {
        // 各トラックのオーディオバッファを準備
        for (const track of tracks) {
          if (track.muted) continue;

          if (!audioBuffersRef.current.has(track.id)) {
            const arrayBuffer = await track.wavData.arrayBuffer();
            const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
            audioBuffersRef.current.set(track.id, audioBuffer);
          }

          const source = audioContextRef.current.createBufferSource();
          const gainNode = audioContextRef.current.createGain();

          source.buffer = audioBuffersRef.current.get(track.id)!;
          gainNode.gain.value = track.volume;

          source.connect(gainNode);
          gainNode.connect(audioContextRef.current.destination);

          // 現在のカーソル位置から再生を開始
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
    }
  };

  // 停止処理
  const handleStop = () => {
    isPlayingRef.current = false;
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    // すべてのソースを停止
    sourcesRef.current.forEach(source => {
      try {
        source.stop();
      } catch (e) {
        console.error('Error stopping source:', e);
      }
    });
    sourcesRef.current = [];
    audioBuffersRef.current.clear();
  };

  // コンポーネントのクリーンアップ
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const handleImportTracks = async () => {
    if (!wavFile || !midFile) return;

    const body: BodyTrainDdspDdspTrainPost = {
      midi_file: midFile,
      wav_file: wavFile,
    };
    try {
      const features = await trainTrigger(body);
      const newTracks: TrackData[] = [];

      for (const feature of features.features) {
        const body: DDSPGenerateParams = {
          z_feature: feature.z_feature,
          loudness: feature.loudness,
          pitch: feature.pitch,
        };
        const response = await generateAudioTrigger(body);
        const wavBlob = new Blob([await response.arrayBuffer()], { type: 'audio/wav' });

        newTracks.push({
          id: `track-${Date.now()}-${Math.random()}`,
          name: `Track ${tracks.length + newTracks.length + 1}`,
          wavData: wavBlob,
          features: feature,
          muted: false,
          volume: 1.0,
        });
      }

      setTracks(prev => [...prev, ...newTracks]);
    } catch (e) {
      console.error('Training error:', e);
    }
    setWavFile(null);
    setMidFile(null);
    closeImportTracksDialog();
  };

  // タイムラインクリック時の処理
  const handleTimelineClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!tracks.length) return;
    if (isPlayingRef.current) {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const totalWidth = Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200);
    const newTime = (clickX / totalWidth) * (tracks[0].wavData.size / (16000 * 2));

    setCurrentTime(newTime);
    playbackStartTimeRef.current = (audioContextRef.current?.currentTime || 0) - newTime;
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {/* 固定ヘッダー (全幅) */}
      <AppBar
        position="fixed"
        sx={{
          top: 0,
          left: 0,
          width: '100vw',
          bgcolor: '#181818',
          zIndex: theme => theme.zIndex.drawer + 1
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Typography variant="h6">DDSP Editor</Typography>

          {/* 再生コントロール */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body1" sx={{ minWidth: '60px' }}>
              {formatTime(currentTime)}
            </Typography>
            <StartButton
              onClick={handlePlay}
              disabled={tracks.length === 0 || isPlaying}
            />
            <StopButton
              onClick={handleStop}
              disabled={!isPlaying}
            />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <RefreshButton onClick={() => setTracks([])} />
            <AddButton disabled={tracks.length !== 0} onClick={openImportTracksDialog} />
          </Box>
        </Toolbar>
      </AppBar>

      {/* ヘッダースペーサー */}
      <Toolbar />

      {/* メインビューポート */}
      <Box sx={{ display: 'flex', height: '100%' }}>
        {/* サイドバー（完全固定） */}
        <Box sx={{
          width: 280,
          position: 'fixed',
          top: 64, // AppBarの高さ
          left: 0,
          height: 'calc(100vh - 64px)',
          zIndex: 5,
          bgcolor: '#1e1e1e',
          display: 'flex',
          flexDirection: 'column',
          borderRight: '1px solid #333',
          mt: '0px',
          pb: 2
        }}>
          {/* タイムライン分の余白 */}
          <Box sx={{ height: '30px', bgcolor: '#1e1e1e', borderBottom: '1px solid #333' }} />
          {/* 時間表示バー */}
          {tracks.map((track, idx) => (
            <TrackSidebar
              key={track.id}
              track={track}
              selected={selectedTrack === track}
              onClick={() => handleTrackClick(track)}
              onMuteToggle={() => handleMuteToggle(track.id)}
              onVolumeChange={(volume) => handleVolumeChange(track.id, volume)}
            />
          ))}
          {/* 空いている部分をクリック可能にする */}
          <Box
            sx={{
              flexGrow: 1,
              cursor: 'pointer',
            }}
            onClick={() => setSelectedTrack(null)}
          />
        </Box>
        {/* 波形部分（横スクロール） */}
        <Box sx={{ flexGrow: 1, overflowX: 'auto', height: 'calc(100vh - 64px)', ml: '280px', bgcolor: 'background.default', position: 'relative' }}>
          {/* 再生位置カーソル */}
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
          <Box onClick={handleTimelineClick} sx={{ cursor: 'pointer' }}>
            {/* タイムラインを追加 */}
            <Timeline
              duration={tracks.length > 0 ? tracks[0].wavData.size / (16000 * 2) : 10}
              width={tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 2000}
            />
            {tracks.map((track, idx) => (
              <TrackRowWaveform
                key={track.id}
                track={track}
                selected={selectedTrack === track}
                setSelectedTrack={setSelectedTrack}
              />
            ))}
            {/* 空いている部分をクリック可能にする */}
            <Box
              sx={{
                height: 'calc(100vh - 64px - 30px - ' + (tracks.length * 100) + 'px)',
                cursor: 'pointer',
              }}
              onClick={() => setSelectedTrack(null)}
            />
          </Box>
        </Box>
      </Box>

      {/* トラックインポートダイアログ */}
      {
        isOpenImportTracksDialog && (
          <ImportTrackDialog
            open={isOpenImportTracksDialog}
            onClose={closeImportTracksDialog}
            wavFile={wavFile}
            setWavFile={setWavFile}
            midFile={midFile}
            setMidFile={setMidFile}
            onImport={handleImportTracks}
          />
        )
      }
      {/* ピアノロール(トラック選択時のみ下部に表示) */}
      {
        selectedTrack && (
          <EditDialog
            currentTime={currentTime}
            selectedTrack={selectedTrack}
            tracks={tracks}
            setTracks={setTracks}
            setSelectedTrack={setSelectedTrack}
            onTimeLineClick={handleTimelineClick}
          />
        )
      }
    </ThemeProvider>
  );
}
