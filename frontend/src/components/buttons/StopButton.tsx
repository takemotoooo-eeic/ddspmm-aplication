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
      style={{ width: 26, height: 26 }}
      disabled={disabled}
    >
      <StopIcon />
    </IconButton>
  );
};
