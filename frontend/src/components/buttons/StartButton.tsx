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
      size="large"
      disabled={disabled}
      sx={{
        fontSize: '2rem',
        '& .MuiSvgIcon-root': {
          fontSize: '2rem',
        },
      }}
    >
      <PlayArrowIcon />
    </IconButton>
  );
};
