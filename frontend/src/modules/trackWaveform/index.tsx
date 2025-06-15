import { Box } from '@mui/material';
import { TrackData } from '../../types/trackData';
import { WaveformDisplay } from './waveDisplay';

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
      width={track.wavData ? Math.floor((track.wavData.size / (16000 * 2)) * 200) : 2000}
      height={70}
    />
  </Box>
); 
