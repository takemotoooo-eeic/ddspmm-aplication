import AddIcon from "@mui/icons-material/Add";
import { IconButton } from "@mui/material";

type Props = {
  onClick: () => void;
  disabled?: boolean;
}

export const AddButton = ({ onClick, disabled }: Props) => {
  return (
    <IconButton
      aria-label="import"
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
      <AddIcon />
    </IconButton>
  );
};
