import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import { IconButton } from "@mui/material";

type Props = {
  onClick: () => void;
  disabled?: boolean;
}

export const StartButton = ({ onClick, disabled }: Props) => {
  return (
    <IconButton
      aria-label="start"
      onClick={onClick}
      style={{ width: 26, height: 26 }}
      disabled={disabled}
    >
      <PlayArrowIcon />
    </IconButton>
  );
};
