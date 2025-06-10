import AddIcon from "@mui/icons-material/Add";
import { IconButton } from "@mui/material";

type Props = {
  onClick: () => void;
}

export const AddButton = ({ onClick }: Props) => {
  return (
    <IconButton
      aria-label="import"
      onClick={onClick}
      style={{ width: 26, height: 26 }}
    >
      <AddIcon />
    </IconButton>
  );
};
