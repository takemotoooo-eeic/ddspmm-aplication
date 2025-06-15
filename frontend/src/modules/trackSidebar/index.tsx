import { Box, Typography } from '@mui/material';
import { MuteButton } from '../../components/buttons/MuteButton';
import { TrackData } from '../../types/trackData';
import { VolumeSlider } from './volumeSlider';

interface TrackRowSidebarProps {
  track: TrackData;
  selected: boolean;
  onClick: () => void;
  onMuteToggle: () => void;
  onVolumeChange: (volume: number) => void;
}

export const TrackSidebar = ({
  track,
  selected,
  onClick,
  onMuteToggle,
  onVolumeChange,
}: TrackRowSidebarProps) => (
  <Box
    sx={{
      width: 280,
      height: 80,
      bgcolor: selected ? '#333' : '#1e1e1e',
      borderRadius: 1,
      display: 'flex',
      alignItems: 'center',
      cursor: 'pointer',
      '&:hover': { bgcolor: '#444' },
      borderBottom: '1px solid #333',
      borderRight: '1px solid #333',
      px: 2,
    }}
    onClick={onClick}
  >
    <Typography sx={{ color: '#fff', minWidth: 120, alignItems: 'center', justifyContent: 'center' }}>{track.name}</Typography>
    <Box sx={{ display: 'flex', gap: 1, ml: 'auto', alignItems: 'center' }}>
      <VolumeSlider
        value={track.volume * 128}
        onChange={(volume) => onVolumeChange(volume / 128)}
      />
      <MuteButton
        isMuted={track.muted}
        onClick={(e) => {
          e.stopPropagation();
          onMuteToggle();
        }}
      />
    </Box>
  </Box>
);
