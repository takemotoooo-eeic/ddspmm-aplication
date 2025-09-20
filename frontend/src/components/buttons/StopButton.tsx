import StopIcon from "@mui/icons-material/Stop";
import { IconButton } from "@mui/material";

type Props = {
  onClick: () => void;
  disabled?: boolean;
}

export const StopButton = ({ onClick, disabled }: Props) => {
  return (
    <IconButton
      aria-label="stop"
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
      <StopIcon />
    </IconButton>
  );
};
