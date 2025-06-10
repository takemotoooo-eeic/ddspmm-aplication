import { Box, Typography } from '@mui/material';
import { MuteButton } from '../../components/buttons/MuteButton';
import { TrackData } from '../../types/trackData';

interface TrackSidebarProps {
  tracks: TrackData[];
  selectedTrackId: string | null;
  onTrackClick: (trackId: string) => void;
  onMuteToggle: (trackId: string) => void;
}

export const TrackSidebar = ({
  tracks,
  selectedTrackId,
  onTrackClick,
  onMuteToggle,
}: TrackSidebarProps) => {
  return (
    <Box
      sx={{
        width: 240,
        bgcolor: 'background.paper',
        p: 2,
        overflowY: 'auto',
        position: 'fixed',
        left: 0,
        top: 64, // AppBarの高さ分
        bottom: 0,
        zIndex: theme => theme.zIndex.drawer + 2,
        borderRight: '1px solid #333',
      }}
    >
      {tracks.map(track => (
        <Box
          key={track.id}
          sx={{
            height: 60,
            bgcolor: selectedTrackId === track.id ? '#333' : '#1e1e1e',
            borderRadius: 1,
            mb: 1,
            display: 'flex',
            alignItems: 'center',
            px: 2,
            cursor: 'pointer',
            '&:hover': { bgcolor: '#444' }
          }}
          onClick={() => onTrackClick(track.id)}
        >
          <Typography sx={{ color: '#fff', minWidth: 80 }}>{track.name}</Typography>
          <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
            <MuteButton
              isMuted={track.muted}
              onClick={() => onMuteToggle(track.id)}
            />
          </Box>
        </Box>
      ))}
    </Box>
  );
}; 
