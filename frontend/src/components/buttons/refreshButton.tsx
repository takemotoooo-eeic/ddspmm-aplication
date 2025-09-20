import RefreshIcon from "@mui/icons-material/Refresh";
import { IconButton } from "@mui/material";

type Props = {
  onClick: () => void;
  disabled?: boolean;
}

export const RefreshButton = ({ onClick, disabled }: Props) => {
  return (
    <IconButton
      aria-label="refresh"
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
      <RefreshIcon />
    </IconButton>
  );
};
