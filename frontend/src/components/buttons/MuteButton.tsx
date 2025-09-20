import VolumeOffIcon from '@mui/icons-material/VolumeOff';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import { IconButton } from '@mui/material';

interface MuteButtonProps {
  isMuted: boolean;
  onClick: (e: React.MouseEvent) => void;
}

export const MuteButton = ({ isMuted, onClick }: MuteButtonProps) => {
  return (
    <IconButton
      onClick={onClick}
      size="large"
      sx={{
        fontSize: '2rem',
        '& .MuiSvgIcon-root': {
          fontSize: '2rem',
        },
        color: isMuted ? '#ff4444' : '#fff',
        '&:hover': {
          color: isMuted ? '#ff6666' : '#ccc',
        },
      }}
    >
      {isMuted ? <VolumeOffIcon /> : <VolumeUpIcon />}
    </IconButton>
  );
}; 
