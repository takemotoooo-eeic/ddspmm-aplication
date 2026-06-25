import {
  AppBar,
  Box,
  createTheme,
  CssBaseline,
  ThemeProvider,
  Toolbar,
  Typography,
} from '@mui/material';
import { useRef, useState } from 'react';
import { editMelodyflow, generateDdspAudio } from './api/backend';
import { ExportButton } from './components/buttons/ExportButton';
import { AddButton } from './components/buttons/ImportButton';
import { LoadButton } from './components/buttons/LoadButton';
import { RefreshButton } from './components/buttons/refreshButton';
import { SettingsButton } from './components/buttons/SettingsButton';
import { StartButton } from './components/buttons/StartButton';
import { StopButton } from './components/buttons/StopButton';
import { DiffusionSettingsDialog } from './components/dialogs/DiffusionSettingsDialog';
import { ImportTrackDialog } from './components/dialogs/ImportTrackDialog';
import { LoadTrackDialog } from './components/dialogs/LoadTrackDialog';
import { TtmEditDialog } from './components/dialogs/TtmEditDialog';
import { DEFAULT_NUM_DENOISING_STEPS } from './components/editors/DiffusionSettingsEditor';
import { EditPanel } from './components/editPanel/EditPanel';
import { TIME_SCALE } from './constants/editor';
import { useAudioPlayback } from './hooks/useAudioPlayback';
import { useDisclosure } from './hooks/useDisclosure';
import { Timeline } from './modules/timeLine';
import { TrackSidebar } from './modules/trackSidebar';
import { TrackRowWaveform } from './modules/trackWaveform';
import { importTracksFromFiles } from './services/importTracks';
import { AppMode } from './types/appMode';
import {
  EDITED_TRACK_NAME,
  isEditedTrack,
  isOriginalTrack,
  supportsParamLoad,
  TrackData,
} from './types/trackData';
import { durationToWidth, wavDurationSec } from './utils/audio';
import { downloadTracksExport, loadTracksFromExportFile } from './utils/exportParams';

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
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [tracks, setTracks] = useState<TrackData[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<TrackData | null>(null);
  const [numDenoisingSteps, setNumDenoisingSteps] = useState(DEFAULT_NUM_DENOISING_STEPS);

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
  const settingsDialog = useDisclosure({});
  const [ttmDialog, setTtmDialog] = useState<{
    startSec: number;
    endSec: number;
    anchor: { top: number; left: number };
  } | null>(null);
  const [ttmLoading, setTtmLoading] = useState(false);
  const waveformScrollRef = useRef<HTMLDivElement>(null);

  const handleModeChange = (mode: AppMode) => {
    setAppMode(mode);
    setTracks([]);
    setSelectedTrack(null);
    setTtmDialog(null);
  };

  const handleMuteToggle = (trackId: string) => {
    setTracks(prev => {
      const target = prev.find(t => t.id === trackId);
      if (!target) return prev;

      if (isOriginalTrack(target)) {
        const willUnmute = target.muted;
        return prev.map(t => {
          if (isOriginalTrack(t)) return { ...t, muted: !willUnmute };
          return { ...t, muted: willUnmute };
        });
      }

      const willUnmute = target.muted;
      return prev.map(t => {
        if (t.id === trackId) return { ...t, muted: !willUnmute };
        if (willUnmute && isOriginalTrack(t)) return { ...t, muted: true };
        return t;
      });
    });
  };

  const handleTrackSelect = (track: TrackData) => {
    if (isOriginalTrack(track)) return;
    setSelectedTrack(track);
  };

  const handleVolumeChange = (trackId: string, volume: number) => {
    setTracks(prev => prev.map(t => (t.id === trackId ? { ...t, volume } : t)));
  };

  const handleImport = async () => {
    if (!wavFile) return;
    if (appMode !== 'ttm' && !midFile) return;
    try {
      const newTracks = await importTracksFromFiles(
        appMode,
        wavFile,
        midFile ?? undefined,
      );
      setTracks(newTracks);
    } catch (e) {
      console.error('Import error:', e);
    }
    setWavFile(null);
    setMidFile(null);
    importDialog.close();
  };

  const handleTtmRegionSelected = (
    _track: TrackData,
    startSec: number,
    endSec: number,
    anchor: { top: number; left: number },
  ) => {
    setTtmDialog({ startSec, endSec, anchor });
  };

  const handleTtmSubmit = async (text: string) => {
    if (!ttmDialog) return;
    const editedTrack = tracks.find(isEditedTrack);
    if (!editedTrack) return;
    setTtmLoading(true);
    try {
      const wavFile = new File([editedTrack.wavData], 'edited.wav', {
        type: 'audio/wav',
      });
      const resultBlob = await editMelodyflow({
        wavFile,
        startSec: ttmDialog.startSec,
        endSec: ttmDialog.endSec,
        text,
      });
      const durationSec = await wavDurationSec(resultBlob);
      setTracks(prev =>
        prev.map(t =>
          t.name === EDITED_TRACK_NAME ? { ...t, wavData: resultBlob, durationSec } : t,
        ),
      );
      setTtmDialog(null);
    } catch (e) {
      console.error('TTM edit error:', e);
    } finally {
      setTtmLoading(false);
    }
  };

  const handleLoadParams = async () => {
    if (!zipFile) return;
    try {
      const newTracks = await loadTracksFromExportFile(zipFile, generateDdspAudio);
      setTracks(newTracks);
    } catch (e) {
      console.error('Load error:', e);
    }
    setZipFile(null);
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
            {/* <ModeSelector mode={appMode} onChange={handleModeChange} /> */}
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
            {appMode === 'diffusion_ddsp' && (
              <SettingsButton onClick={settingsDialog.open} />
            )}
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
                  onClick={() => downloadTracksExport(tracks)}
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

      <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
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
              onClick={() => handleTrackSelect(track)}
              onMuteToggle={() => handleMuteToggle(track.id)}
              onVolumeChange={v => handleVolumeChange(track.id, v)}
            />
          ))}
        </Box>

        <Box
          ref={waveformScrollRef}
          sx={{
            flexGrow: 1,
            minWidth: 0,
            overflowX: 'auto',
            overflowY: 'hidden',
            height: 'calc(100vh - 64px)',
            ml: '280px',
            bgcolor: 'background.default',
          }}
        >
          <Box
            onClick={e => handleTimelineClick(e)}
            sx={{
              position: 'relative',
              cursor: 'pointer',
              width: waveformWidth,
              minWidth: waveformWidth,
              flexShrink: 0,
            }}
          >
            <Box
              sx={{
                position: 'absolute',
                left:
                  trackDuration > 0
                    ? (currentTime / trackDuration) * waveformWidth
                    : 0,
                top: 0,
                height: '100%',
                width: 2,
                bgcolor: 'rgba(255,255,255,0.3)',
                zIndex: 2,
                pointerEvents: 'none',
              }}
            />
            <Timeline duration={trackDuration || 10} width={waveformWidth} />
            {tracks.map(track => (
              <TrackRowWaveform
                key={track.id}
                track={track}
                selected={selectedTrack?.id === track.id}
                setSelectedTrack={handleTrackSelect}
                appMode={appMode}
                scrollContainerRef={
                  appMode === 'ttm' ? waveformScrollRef : undefined
                }
                onRegionSelected={
                  appMode === 'ttm' ? handleTtmRegionSelected : undefined
                }
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
          wavOnly={appMode === 'ttm'}
        />
      )}

      {ttmDialog && (
        <TtmEditDialog
          open
          startSec={ttmDialog.startSec}
          endSec={ttmDialog.endSec}
          anchor={ttmDialog.anchor}
          loading={ttmLoading}
          onClose={() => !ttmLoading && setTtmDialog(null)}
          onSubmit={handleTtmSubmit}
        />
      )}

      {loadDialog.isOpen && supportsParamLoad(appMode) && (
        <LoadTrackDialog
          open={loadDialog.isOpen}
          onClose={loadDialog.close}
          zipFile={zipFile}
          setZipFile={setZipFile}
          onLoad={handleLoadParams}
        />
      )}

      {appMode === 'diffusion_ddsp' && (
        <DiffusionSettingsDialog
          open={settingsDialog.isOpen}
          onClose={settingsDialog.close}
          numDenoisingSteps={numDenoisingSteps}
          onNumDenoisingStepsChange={setNumDenoisingSteps}
        />
      )}

      {selectedTrack && !isOriginalTrack(selectedTrack) && appMode !== 'ttm' && (
        <EditPanel
          appMode={appMode}
          currentTime={currentTime}
          selectedTrack={selectedTrack}
          tracks={tracks}
          setTracks={setTracks}
          setSelectedTrack={setSelectedTrack}
          onTimeLineClick={e => handleTimelineClick(e, true)}
          numDenoisingSteps={numDenoisingSteps}
        />
      )}
    </ThemeProvider>
  );
};
