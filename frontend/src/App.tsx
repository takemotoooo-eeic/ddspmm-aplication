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
import { RefreshButton } from './components/buttons/refreshButton';
import { useDisclosure } from './hooks/useDisclosure';
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
    setWavFile(null);
    setMidFile(null);
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
          <Typography variant="h6">DDSP Editor</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* リロードボタン */}
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
          zIndex: 20,
          bgcolor: '#1e1e1e',
          display: 'flex',
          flexDirection: 'column',
          borderRight: '1px solid #333',
          mt: '0px', // 余白はBoxで表現
          pb: 2
        }}>
          {/* タイムライン分の余白 */}
          <Box sx={{ height: '30px', bgcolor: '#1e1e1e', borderBottom: '1px solid #333' }} />
          {/* 時間表示バー */}
          {tracks.map((track, idx) => (
            <TrackSidebar
              key={track.id}
              track={track}
              selected={selectedTrack === track.id}
              onClick={() => handleTrackClick(track.id)}
              onMuteToggle={() => handleMuteToggle(track.id)}
              onVolumeChange={(volume) => handleVolumeChange(track.id, volume)}
            />
          ))}
        </Box>
        {/* 波形部分（横スクロール） */}
        <Box sx={{ flexGrow: 1, overflowX: 'auto', height: 'calc(100vh - 64px)', ml: '280px', bgcolor: 'background.default' }}>
          <Box>
            {/* タイムラインを追加 */}
            <Timeline
              duration={tracks.length > 0 ? tracks[0].wavData.size / (16000 * 2) : 10}
              width={tracks.length > 0 ? Math.floor((tracks[0].wavData.size / (16000 * 2)) * 200) : 2000}
            />
            {tracks.map((track, idx) => (
              <TrackRowWaveform
                key={track.id}
                track={track}
              />
            ))}
          </Box>
        </Box>
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
    </ThemeProvider>
  );
}
