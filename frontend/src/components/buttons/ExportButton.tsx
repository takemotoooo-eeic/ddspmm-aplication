import FileDownloadIcon from '@mui/icons-material/FileDownload';
import { IconButton, Tooltip } from '@mui/material';

type Props = {
  onClick: () => void;
  disabled?: boolean;
};

export const ExportButton = ({ onClick, disabled }: Props) => (
  <Tooltip title="Export synthesis parameters (JSONL)">
    <span>
      <IconButton
        aria-label="export parameters"
        onClick={onClick}
        size="large"
        disabled={disabled}
        sx={{
          '& .MuiSvgIcon-root': { fontSize: '2rem' },
        }}
      >
        <FileDownloadIcon />
      </IconButton>
    </span>
  </Tooltip>
);
