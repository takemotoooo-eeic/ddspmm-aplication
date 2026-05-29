import SettingsIcon from '@mui/icons-material/Settings';
import { IconButton } from '@mui/material';

type Props = {
  onClick: () => void;
  disabled?: boolean;
};

export const SettingsButton = ({ onClick, disabled }: Props) => {
  return (
    <IconButton
      aria-label="settings"
      onClick={onClick}
      size="large"
      disabled={disabled}
      sx={{
        fontSize: '2rem',
        '& .MuiSvgIcon-root': {
          fontSize: '2rem',
        },
      }}
    >
      <SettingsIcon />
    </IconButton>
  );
};
