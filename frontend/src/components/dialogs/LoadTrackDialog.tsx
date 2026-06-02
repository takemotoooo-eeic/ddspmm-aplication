import { Box, Button, CircularProgress, Dialog, Typography } from '@mui/material';
import { useState } from 'react';

interface LoadTrackDialogProps {
  open: boolean;
  onClose: () => void;
  zipFile: File | null;
  setZipFile: (file: File | null) => void;
  onLoad: () => Promise<void>;
}

export const LoadTrackDialog = ({
  open,
  onClose,
  zipFile,
  setZipFile,
  onLoad,
}: LoadTrackDialogProps) => {
  const [isLoading, setIsLoading] = useState(false);

  return (
    <Dialog open={open} onClose={() => !isLoading && onClose()}>
      <Box sx={{ bgcolor: 'background.paper', p: 3, minWidth: 400 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          LOAD SYNTHESIS PARAMETERS
        </Typography>
        <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
          Select ZIP File
          <input
            type="file"
            accept=".zip,application/zip"
            hidden
            onChange={e => setZipFile(e.target.files?.[0] ?? null)}
          />
        </Button>
        {zipFile && (
          <Typography variant="body2" sx={{ mb: 2 }}>
            Selected: {zipFile.name}
          </Typography>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            variant="contained"
            sx={{ ml: 2 }}
            onClick={async () => {
              setIsLoading(true);
              try {
                await onLoad();
              } finally {
                setIsLoading(false);
              }
            }}
            disabled={!zipFile || isLoading}
          >
            {isLoading ? <CircularProgress size={16} color="inherit" /> : 'Load'}
          </Button>
        </Box>
      </Box>
    </Dialog>
  );
};
