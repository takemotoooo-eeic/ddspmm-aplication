import { Box } from '@mui/material';
import { TrackData } from '../types/trackData';
import { WaveformDisplay } from './trackWaveform/waveDisplay/WaveformDisplay';

interface TrackRowWaveformProps {
  track: TrackData;
}

export const TrackRowWaveform = ({ track }: TrackRowWaveformProps) => (
  <Box
    sx={{
      height: 80,
      bgcolor: '#222',
      borderRadius: 1,
      overflow: 'hidden',
      borderBottom: '1px solid #333',
      display: 'flex',
      alignItems: 'center',
    }}
  >
    <WaveformDisplay
      wavData={track.wavData}
      width={2000}
      height={70}
    />
  </Box>
); 
