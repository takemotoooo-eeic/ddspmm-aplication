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
      style={{ width: 26, height: 26 }}
      disabled={disabled}
    >
      <RefreshIcon />
    </IconButton>
  );
};
