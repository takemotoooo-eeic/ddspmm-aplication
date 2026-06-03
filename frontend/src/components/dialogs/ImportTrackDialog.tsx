import { Box, Button, CircularProgress, Dialog, Typography } from '@mui/material';
import { useState } from 'react';

interface ImportTrackDialogProps {
  open: boolean;
  onClose: () => void;
  wavFile: File | null;
  setWavFile: (file: File | null) => void;
  midFile: File | null;
  setMidFile: (file: File | null) => void;
  onImport: () => Promise<void>;
  /** TTM モードなど MIDI 不要のとき true */
  wavOnly?: boolean;
}

export const ImportTrackDialog = ({
  open,
  onClose,
  wavFile,
  setWavFile,
  midFile,
  setMidFile,
  onImport,
  wavOnly = false,
}: ImportTrackDialogProps) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleImport = async () => {
    setIsLoading(true);
    try {
      await onImport();
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onClose={() => !isLoading && onClose()}>
      <Box sx={{ bgcolor: 'background.paper', p: 3, minWidth: 400 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          IMPORT TRACKS
        </Typography>
        <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
          Select WAV File
          <input
            type="file"
            accept="audio/wav"
            hidden
            onChange={e => setWavFile(e.target.files?.[0] ?? null)}
          />
        </Button>
        {wavFile && (
          <Typography variant="body2" sx={{ mb: 2 }}>
            Selected: {wavFile.name}
          </Typography>
        )}
        {!wavOnly && (
          <>
            <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
              Select MIDI File
              <input
                type="file"
                accept="audio/midi,.mid"
                hidden
                onChange={e => setMidFile(e.target.files?.[0] ?? null)}
              />
            </Button>
            {midFile && (
              <Typography variant="body2" sx={{ mb: 2 }}>
                Selected: {midFile.name}
              </Typography>
            )}
          </>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            variant="contained"
            sx={{ ml: 2 }}
            onClick={handleImport}
            disabled={!wavFile || (!wavOnly && !midFile) || isLoading}
          >
            {isLoading ? <CircularProgress size={16} color="inherit" /> : 'Import'}
          </Button>
        </Box>
      </Box>
    </Dialog>
  );
};
