import { Box } from '@mui/material';
import { TIME_SCALE } from '../../constants/editor';
import { TrackData } from '../../types/trackData';
import { blobDurationSec, durationToWidth } from '../../utils/audio';
import { WaveformDisplay } from './waveDisplay';

interface TrackRowWaveformProps {
  track: TrackData;
  setSelectedTrack: (track: TrackData) => void;
  selected: boolean;
}

export const TrackRowWaveform = ({ track, setSelectedTrack, selected }: TrackRowWaveformProps) => {
  const durationSec = track.wavData ? blobDurationSec(track.wavData) : 0;
  const width = Math.max(1, durationToWidth(durationSec, TIME_SCALE));

  return (
    <Box
      sx={{
        height: 80,
        bgcolor: selected ? '#333' : '#222',
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
      />
    </Box>
  );
}; 
