import {
  AppBar,
  Box,
  createTheme,
  CssBaseline,
  ThemeProvider,
  Toolbar,
  Typography,
} from '@mui/material';
import { useState } from 'react';
import { generateDdspAudio } from './api/backend';
import { ExportButton } from './components/buttons/ExportButton';
import { AddButton } from './components/buttons/ImportButton';
import { LoadButton } from './components/buttons/LoadButton';
import { RefreshButton } from './components/buttons/refreshButton';
import { StartButton } from './components/buttons/StartButton';
import { StopButton } from './components/buttons/StopButton';
import { ImportTrackDialog } from './components/dialogs/ImportTrackDialog';
import { LoadTrackDialog } from './components/dialogs/LoadTrackDialog';
import { EditPanel } from './components/editPanel/EditPanel';
import { ModeSelector } from './components/layout/ModeSelector';
import { TIME_SCALE } from './constants/editor';
import { useAudioPlayback } from './hooks/useAudioPlayback';
import { useDisclosure } from './hooks/useDisclosure';
import { Timeline } from './modules/timeLine';
import { TrackSidebar } from './modules/trackSidebar';
import { TrackRowWaveform } from './modules/trackWaveform';
import { DDSPGenerateParams } from './orval/models/backend-api';
import { importTracksFromFiles } from './services/importTracks';
import { AppMode } from './types/appMode';
import { supportsParamLoad, TrackData } from './types/trackData';
import { durationToWidth } from './utils/audio';
import { downloadTracksAsJsonl } from './utils/exportParams';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#646cff' },
    background: { default: '#242424', paper: '#1e1e1e' },
    text: { primary: '#ffffff' },
  },
});

const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
};

export default function App() {
  const [appMode, setAppMode] = useState<AppMode>('diffusion_ddsp');
  const [wavFile, setWavFile] = useState<File | null>(null);
  const [midFile, setMidFile] = useState<File | null>(null);
  const [jsonlFile, setJsonlFile] = useState<File | null>(null);
  const [tracks, setTracks] = useState<TrackData[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<TrackData | null>(null);

  const {
    isPlaying,
    currentTime,
    trackDuration,
    handlePlay,
    handleStop,
    seekToTime,
    isPlayingRef,
  } = useAudioPlayback(tracks);

  const waveformWidth =
    tracks.length > 0 ? durationToWidth(trackDuration, TIME_SCALE) : 2000;

  const importDialog = useDisclosure({});
  const loadDialog = useDisclosure({});

  const handleModeChange = (mode: AppMode) => {
    setAppMode(mode);
    setTracks([]);
    setSelectedTrack(null);
  };

  const handleMuteToggle = (trackId: string) => {
    setTracks(prev =>
      prev.map(t => (t.id === trackId ? { ...t, muted: !t.muted } : t)),
    );
  };

  const handleVolumeChange = (trackId: string, volume: number) => {
    setTracks(prev => prev.map(t => (t.id === trackId ? { ...t, volume } : t)));
  };

  const handleImport = async () => {
    if (!wavFile || !midFile) return;
    try {
      const newTracks = await importTracksFromFiles(appMode, wavFile, midFile);
      setTracks(newTracks);
    } catch (e) {
      console.error('Import error:', e);
    }
    setWavFile(null);
    setMidFile(null);
    importDialog.close();
  };

  const handleLoadParams = async () => {
    if (!jsonlFile) return;
    const text = await jsonlFile.text();
    const lines = text.split('\n');
    const newTracks: TrackData[] = [];
    for (const line of lines) {
      if (!line.trim()) continue;
      const data = JSON.parse(line);
      const body: DDSPGenerateParams = {
        z_feature: data.z_feature,
        loudness: data.loudness,
        pitch: data.pitch,
      };
      const wav = await generateDdspAudio(body);
      const blockSize = 512;
      newTracks.push({
        id: `track-${Date.now()}-${Math.random()}`,
        name: data.instrument_name,
        instrument: data.instrument_name,
        wavData: wav,
        features: data,
        signalLength: data.pitch.length * blockSize,
        muted: false,
        volume: 1,
      });
    }
    setTracks(newTracks);
    setJsonlFile(null);
    loadDialog.close();
  };

  const handleTimelineClick = (
    event: React.MouseEvent<HTMLDivElement>,
    useEditScale = false,
  ) => {
    if (!tracks.length || isPlayingRef.current) return;
    const scale = useEditScale ? TIME_SCALE : 200;
    const rect = event.currentTarget.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const width = durationToWidth(trackDuration, scale);
    seekToTime((clickX / width) * trackDuration);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: '100vw',
          bgcolor: '#181818',
          zIndex: t => t.zIndex.drawer + 1,
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ModeSelector mode={appMode} onChange={handleModeChange} />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h5" sx={{ minWidth: 60 }}>
              {formatTime(currentTime)}
            </Typography>
            <StartButton
              onClick={handlePlay}
              disabled={tracks.length === 0 || isPlaying}
            />
            <StopButton onClick={handleStop} disabled={!isPlaying} />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <RefreshButton
              onClick={() => {
                setTracks([]);
                setSelectedTrack(null);
              }}
            />
            {supportsParamLoad(appMode) && (
              <>
                <LoadButton
                  disabled={tracks.length !== 0}
                  onClick={loadDialog.open}
                />
                <ExportButton
                  disabled={
                    tracks.length === 0 ||
                    !tracks.some(t => t.features != null)
                  }
                  onClick={() => downloadTracksAsJsonl(tracks)}
                />
              </>
            )}
            <AddButton
              disabled={tracks.length !== 0}
              onClick={importDialog.open}
            />
          </Box>
        </Toolbar>
      </AppBar>

      <Toolbar />

      <Box sx={{ display: 'flex', height: '100%' }}>
        <Box
          sx={{
            width: 280,
            position: 'fixed',
            top: 64,
            left: 0,
            height: 'calc(100vh - 64px)',
            zIndex: 5,
            bgcolor: '#1e1e1e',
            borderRight: '1px solid #333',
          }}
        >
          <Box sx={{ height: 30, borderBottom: '1px solid #333' }} />
          {tracks.map(track => (
            <TrackSidebar
              key={track.id}
              track={track}
              selected={selectedTrack?.id === track.id}
              onClick={() => setSelectedTrack(track)}
              onMuteToggle={() => handleMuteToggle(track.id)}
              onVolumeChange={v => handleVolumeChange(track.id, v)}
            />
          ))}
        </Box>

        <Box
          sx={{
            flexGrow: 1,
            overflowX: 'auto',
            height: 'calc(100vh - 64px)',
            ml: '280px',
            bgcolor: 'background.default',
            position: 'relative',
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              left:
                trackDuration > 0 ? (currentTime / trackDuration) * waveformWidth : 0,
              top: 0,
              height: '100%',
              width: 2,
              bgcolor: 'rgba(255,255,255,0.3)',
              zIndex: 2,
              pointerEvents: 'none',
            }}
          />
          <Box onClick={e => handleTimelineClick(e)} sx={{ cursor: 'pointer' }}>
            <Timeline duration={trackDuration || 10} width={waveformWidth} />
            {tracks.map(track => (
              <TrackRowWaveform
                key={track.id}
                track={track}
                selected={selectedTrack?.id === track.id}
                setSelectedTrack={setSelectedTrack}
              />
            ))}
          </Box>
        </Box>
      </Box>

      {importDialog.isOpen && (
        <ImportTrackDialog
          open={importDialog.isOpen}
          onClose={importDialog.close}
          wavFile={wavFile}
          setWavFile={setWavFile}
          midFile={midFile}
          setMidFile={setMidFile}
          onImport={handleImport}
        />
      )}

      {loadDialog.isOpen && supportsParamLoad(appMode) && (
        <LoadTrackDialog
          open={loadDialog.isOpen}
          onClose={loadDialog.close}
          jsonlFile={jsonlFile}
          setJsonlFile={setJsonlFile}
          onLoad={handleLoadParams}
        />
      )}

      {selectedTrack && (
        <EditPanel
          appMode={appMode}
          currentTime={currentTime}
          selectedTrack={selectedTrack}
          tracks={tracks}
          setTracks={setTracks}
          setSelectedTrack={setSelectedTrack}
          onTimeLineClick={e => handleTimelineClick(e, true)}
        />
      )}
    </ThemeProvider>
  );
};
