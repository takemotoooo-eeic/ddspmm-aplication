import { Box, Button, CircularProgress, Dialog, Typography } from '@mui/material';
import React, { useState } from 'react';

interface LoadTrackDialogProps {
  open: boolean;
  onClose: () => void;
  jsonlFile: File | null;
  setJsonlFile: (file: File | null) => void;
  onLoad: () => void;
}

export const LoadTrackDialog: React.FC<LoadTrackDialogProps> = ({
  open,
  onClose,
  jsonlFile,
  setJsonlFile,
  onLoad,
}) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleLoad = async () => {
    setIsLoading(true);
    try {
      await onLoad();
    } finally {
      setIsLoading(false);
    }
  };

  const isLoadEnabled = jsonlFile && !isLoading;

  return (
    <Dialog open={open} onClose={() => !isLoading && onClose()}>
      <Box sx={{ bgcolor: 'background.paper', p: 3, minWidth: 400 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          LOAD SYNTHESIS PARAMETERS
        </Typography>

        {/* JSONL ファイル選択 */}
        <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
          Select JSONL File
          <input
            type="file"
            accept=".jsonl"
            hidden
            onChange={e => {
              const file = e.target.files?.[0] || null;
              setJsonlFile(file);
            }}
          />
        </Button>
        {jsonlFile && (
          <Typography variant="body2" sx={{ mb: 2 }}>
            Selected: {jsonlFile.name}
          </Typography>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button onClick={onClose} disabled={isLoading}>Cancel</Button>
          <Button
            variant="contained"
            color="primary"
            sx={{ ml: 2 }}
            onClick={handleLoad}
            disabled={!isLoadEnabled}
          >
            {isLoading ? (
              <CircularProgress size={16} color="inherit" />
            ) : (
              'Load'
            )}
          </Button>
        </Box>
      </Box>
    </Dialog>
  );
};
