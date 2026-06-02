import { Box } from '@mui/material';
import { TIME_SCALE } from '../../constants/editor';
import { TrackData, isOriginalTrack, trackDurationSec } from '../../types/trackData';
import { durationToWidth } from '../../utils/audio';
import { WaveformDisplay } from './waveDisplay';

interface TrackRowWaveformProps {
  track: TrackData;
  setSelectedTrack: (track: TrackData) => void;
  selected: boolean;
}

export const TrackRowWaveform = ({ track, setSelectedTrack, selected }: TrackRowWaveformProps) => {
  const durationSec = track.wavData ? trackDurationSec(track) : 0;
  const width = Math.max(1, durationToWidth(durationSec, TIME_SCALE));
  const isOriginal = isOriginalTrack(track);

  return (
    <Box
      sx={{
        height: 80,
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
        selectable={!isOriginal}
      />
    </Box>
  );
}; 
