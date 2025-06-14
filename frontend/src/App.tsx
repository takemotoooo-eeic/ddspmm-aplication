import {
  AppBar,
  Box,
  Button,
  createTheme,
  CssBaseline,
  Dialog,
  ThemeProvider,
  Toolbar,
  Typography
} from '@mui/material';
import { useState } from 'react';
import { AddButton } from './components/buttons/ImportButton';
import { WaveformDisplay } from './components/wavDisplay/WaveformDisplay';
import { useDisclosure } from './hooks/useDisclosure';
import { TrackSidebar } from './modules/trackSidebar';
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
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null);

  const handleTrackClick = (trackId: string) => {
    setSelectedTrack(trackId);
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

    closeImportTracksDialog();
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
          <Typography variant="h6">DDSPMM Editor</Typography>
          <AddButton onClick={openImportTracksDialog} />
        </Toolbar>
      </AppBar>

      {/* ヘッダースペーサー */}
      <Toolbar />

      {/* サイドバー：トラック一覧（固定表示） */}
      <TrackSidebar
        tracks={tracks}
        selectedTrackId={selectedTrack}
        onTrackClick={handleTrackClick}
        onMuteToggle={handleMuteToggle}
        onVolumeChange={handleVolumeChange}
      />

      {/* メインビューポート */}
      <Box
        sx={{
          display: 'flex',
          flexGrow: 1,
          height: `calc(100vh - 64px)`, // AppBar の高さ分を引く
          bgcolor: 'background.default',
        }}
      >
        {/* タイムライン＋コントロールパネル */}
        <Box sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1, ml: '240px' }}>
          {/* 上部：タイムライン */}
          <Box
            sx={{
              flexGrow: 1,
              bgcolor: 'background.default',
              p: 2,
              overflowX: 'auto', // 横スクロール
              minWidth: 1200,    // 必要に応じて調整
            }}
          >
            {/* 各トラックの波形を縦に並べて表示 */}
            {tracks.map((track, idx) => (
              <Box key={track.id} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box sx={{ width: 2000, height: 60, bgcolor: '#222', borderRadius: 1, overflow: 'hidden' }}>
                  <WaveformDisplay
                    wavData={track.wavData}
                    width={2000}
                    isLastTrack={idx === tracks.length - 1}
                  />
                </Box>
              </Box>
            ))}
          </Box>

          {/* 下部：コントロールパネル */}
        </Box>

        {/* トラックインポートダイアログ */}
        {
          isOpenImportTracksDialog && (
            <Dialog open={isOpenImportTracksDialog} onClose={closeImportTracksDialog}>
              <Box sx={{ bgcolor: 'background.paper', p: 3, minWidth: 400 }}>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  IMPORT TRACKS
                </Typography>
                {/* WAV ファイル選択 */}
                <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
                  Select WAV File
                  <input
                    type="file"
                    accept="audio/wav"
                    hidden
                    onChange={e => {
                      const file = e.target.files?.[0] || null;
                      setWavFile(file);
                    }}
                  />
                </Button>
                {wavFile && (
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    Selected: {wavFile.name}
                  </Typography>
                )}

                {/* MIDI ファイル選択 */}
                <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
                  Select MIDI File
                  <input
                    type="file"
                    accept="audio/midi, .mid"
                    hidden
                    onChange={e => {
                      const file = e.target.files?.[0] || null;
                      setMidFile(file);
                    }}
                  />
                </Button>
                {midFile && (
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    Selected: {midFile.name}
                  </Typography>
                )}
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                  <Button onClick={closeImportTracksDialog}>Cancel</Button>
                  <Button
                    variant="contained"
                    color="primary"
                    sx={{ ml: 2 }}
                    onClick={handleImportTracks}
                  >
                    Import
                  </Button>
                </Box>
              </Box>
            </Dialog>
          )
        }
      </Box>
    </ThemeProvider>
  );
}
