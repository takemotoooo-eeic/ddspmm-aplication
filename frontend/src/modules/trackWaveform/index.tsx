import { Box } from '@mui/material';
import { TIME_SCALE } from '../../constants/editor';
import { AppMode } from '../../types/appMode';
import { TrackData, isEditedTrack, isOriginalTrack, trackDurationSec } from '../../types/trackData';
import { durationToWidth } from '../../utils/audio';
import { WaveformDisplay } from './waveDisplay';

interface TrackRowWaveformProps {
  track: TrackData;
  setSelectedTrack: (track: TrackData) => void;
  selected: boolean;
  appMode?: AppMode;
  scrollContainerRef?: React.RefObject<HTMLElement | null>;
  onRegionSelected?: (
    track: TrackData,
    startSec: number,
    endSec: number,
    anchor: { top: number; left: number },
  ) => void;
}

export const TrackRowWaveform = ({
  track,
  setSelectedTrack,
  selected,
  appMode,
  scrollContainerRef,
  onRegionSelected,
}: TrackRowWaveformProps) => {
  const durationSec = track.wavData ? trackDurationSec(track) : 0;
  const width = Math.max(1, durationToWidth(durationSec, TIME_SCALE));
  const isOriginal = isOriginalTrack(track);
  const regionSelectEnabled =
    appMode === 'ttm' && isEditedTrack(track) && onRegionSelected != null;

  return (
    <Box
      sx={{
        height: 80,
        width,
        minWidth: width,
        flexShrink: 0,
        bgcolor: isOriginal ? '#2a2838' : selected ? '#333' : '#222',
        borderRadius: 1,
        overflow: 'hidden',
        borderBottom: '1px solid #333',
        display: 'flex',
        alignItems: 'center',
      }}
    >
      <WaveformDisplay
        wavData={track.wavData}
        width={width}
        height={70}
        track={track}
        setSelectedTrack={setSelectedTrack}
        backgroundColor={isOriginal ? '#2a2838' : '#1e1e1e'}
        trackColor={isOriginal ? '#8b8ba8' : '#646cff'}
        selectable={!isOriginal && !regionSelectEnabled}
        regionSelectEnabled={regionSelectEnabled}
        durationSec={durationSec}
        scrollContainerRef={regionSelectEnabled ? scrollContainerRef : undefined}
        onRegionSelected={
          onRegionSelected
            ? (startSec, endSec, anchor) =>
              onRegionSelected(track, startSec, endSec, anchor)
            : undefined
        }
      />
    </Box>
  );
}; 
