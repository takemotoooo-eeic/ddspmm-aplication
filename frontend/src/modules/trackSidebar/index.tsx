import { Box, Typography } from '@mui/material';
import { MuteButton } from '../../components/buttons/MuteButton';
import { TrackData } from '../../types/trackData';
import { VolumeSlider } from './volumeSlider';

interface TrackSidebarProps {
  tracks: TrackData[];
  selectedTrackId: string | null;
  onTrackClick: (trackId: string) => void;
  onMuteToggle: (trackId: string) => void;
  onVolumeChange: (trackId: string, volume: number) => void;
}

export const TrackSidebar = ({
  tracks,
  selectedTrackId,
  onTrackClick,
  onMuteToggle,
  onVolumeChange,
}: TrackSidebarProps) => {
  return (
    <Box
      sx={{
        width: 240,
        bgcolor: 'background.paper',
        pt: 2,
        pb: 2,
        overflowY: 'auto',
        position: 'fixed',
        left: 0,
        top: 64, // AppBarの高さ分
        bottom: 0,
        zIndex: theme => theme.zIndex.drawer + 2,
        borderRight: '1px solid #333',
      }}
    >
      {tracks.map((track, idx) => (
        <Box
          key={track.id}
          sx={{
            height: 60,
            bgcolor: selectedTrackId === track.id ? '#333' : '#1e1e1e',
            borderRadius: 1,
            mb: 1,
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            '&:hover': { bgcolor: '#444' },
            borderBottom: '1px solid #333',
            px: 0,
          }}
          onClick={() => onTrackClick(track.id)}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', px: 2 }}>
            <Typography sx={{ color: '#fff', minWidth: 80 }}>{track.name}</Typography>
            <Box sx={{ display: 'flex', gap: 1, ml: 'auto', alignItems: 'center' }}>
              <VolumeSlider
                value={track.volume * 128}
                onChange={(volume) => onVolumeChange(track.id, volume / 128)}
              />
              <MuteButton
                isMuted={track.muted}
                onClick={(e) => {
                  e.stopPropagation();
                  onMuteToggle(track.id);
                }}
              />
            </Box>
          </Box>
        </Box>
      ))}
    </Box>
  );
}; 
