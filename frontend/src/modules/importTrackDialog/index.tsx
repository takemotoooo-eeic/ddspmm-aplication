import { Box, Button, Dialog, Typography } from '@mui/material';
import React from 'react';

interface ImportTrackDialogProps {
  open: boolean;
  onClose: () => void;
  wavFile: File | null;
  setWavFile: (file: File | null) => void;
  midFile: File | null;
  setMidFile: (file: File | null) => void;
  onImport: () => void;
}

export const ImportTrackDialog: React.FC<ImportTrackDialogProps> = ({
  open,
  onClose,
  wavFile,
  setWavFile,
  midFile,
  setMidFile,
  onImport
}) => {
  return (
    <Dialog open={open} onClose={onClose}>
      <Box sx={{ bgcolor: 'background.paper', p: 3, minWidth: 400 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          IMPORT TRACKS
        </Typography>
        {/* WAV ファイル選択 */}
        <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
          Select WAV File
          <input
            type="file"
            accept="audio/wav"
            hidden
            onChange={e => {
              const file = e.target.files?.[0] || null;
              setWavFile(file);
            }}
          />
        </Button>
        {wavFile && (
          <Typography variant="body2" sx={{ mb: 2 }}>
            Selected: {wavFile.name}
          </Typography>
        )}

        {/* MIDI ファイル選択 */}
        <Button component="label" variant="outlined" fullWidth sx={{ mb: 2 }}>
          Select MIDI File
          <input
            type="file"
            accept="audio/midi, .mid"
            hidden
            onChange={e => {
              const file = e.target.files?.[0] || null;
              setMidFile(file);
            }}
          />
        </Button>
        {midFile && (
          <Typography variant="body2" sx={{ mb: 2 }}>
            Selected: {midFile.name}
          </Typography>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            variant="contained"
            color="primary"
            sx={{ ml: 2 }}
            onClick={onImport}
          >
            Import
          </Button>
        </Box>
      </Box>
    </Dialog>
  );
};
