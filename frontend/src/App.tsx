import {
  AppBar,
  Box,
  Button,
  createTheme,
  CssBaseline,
  Dialog,
  Drawer,
  ThemeProvider,
  Toolbar,
  Typography
} from '@mui/material';
import { useState } from 'react';
import { AddButton } from './components/buttons/ImportButton';
import { useDisclosure } from './hooks/useDisclosure';
import { useTrainDdsp } from './orval/backend-api';
import { BodyTrainDdspDdspTrainPost } from './orval/models/backend-api';

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
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null);

  const handleTrackClick = (trackId: string) => {
    setSelectedTrack(trackId);
  };
  const { trigger: trainTrigger } = useTrainDdsp();
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
      console.log('Received features:', features);
    } catch (e) {
      console.error('Training error:', e);
    } finally {
      closeImportTracksDialog();
    }
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

      {/* メインビューポート */}
      <Box
        sx={{
          display: 'flex',
          flexGrow: 1,
          height: `calc(100vh - 64px)`, // AppBar の高さ分を引く
          bgcolor: 'background.default'
        }}
      >
        {/* サイドバー：トラック一覧 */}
        <Box
          sx={{
            width: 240,
            bgcolor: 'background.paper',
            p: 2,
            overflowY: 'auto'
          }}
        >
          {['Track 1', 'Track 2', 'Track 3'].map(track => (
            <Box
              key={track}
              sx={{
                height: 60,
                bgcolor: '#1e1e1e',
                borderRadius: 1,
                mb: 1,
                display: 'flex',
                alignItems: 'center',
                px: 2,
                cursor: 'pointer',
                '&:hover': { bgcolor: '#333' }
              }}
              onClick={() => handleTrackClick(track)}
            >
              <Typography sx={{ color: '#fff' }}>{track}</Typography>
            </Box>
          ))}
        </Box>

        {/* タイムライン＋コントロールパネル */}
        <Box sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1 }}>
          {/* 上部：タイムライン */}
          <Box
            sx={{
              flexGrow: 1,
              bgcolor: 'background.default',
              p: 2,
              overflow: 'auto'
            }}
          >
            {/* 波形/オートメーション */}
          </Box>

          {/* 下部：コントロールパネル */}
          <Box
            sx={{
              height: 200,
              bgcolor: 'background.paper',
              p: 2,
              borderTop: '1px solid #333'
            }}
          >
            {/* コントロールモジュール */}
          </Box>
        </Box>

        {/* 編集ウィンドウ（Drawer） */}
        <Drawer
          anchor="right"
          open={Boolean(selectedTrack)}
          onClose={() => setSelectedTrack(null)}
        >
          <Box sx={{ width: 320, bgcolor: 'background.paper', p: 3, height: '100%' }}>
            <Typography variant="h6" sx={{ mb: 3 }}>
              {selectedTrack} Editor
            </Typography>
            {/* エディタコンテンツ */}
          </Box>
        </Drawer>

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
