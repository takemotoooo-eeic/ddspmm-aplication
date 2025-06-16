import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Box, Button, CircularProgress, Collapse, Dialog, IconButton, LinearProgress, TextField, Typography } from '@mui/material';
import React, { useState } from 'react';
import { LearnData } from '../../types/learnData';

interface ImportTrackDialogProps {
  open: boolean;
  onClose: () => void;
  wavFile: File | null;
  setWavFile: (file: File | null) => void;
  midFile: File | null;
  setMidFile: (file: File | null) => void;
  onImport: (epochs: number, lr: number) => void;
  learnData: LearnData | null;
  setLearnData: (data: LearnData | null) => void;
}

export const ImportTrackDialog: React.FC<ImportTrackDialogProps> = ({
  open,
  onClose,
  wavFile,
  setWavFile,
  midFile,
  setMidFile,
  onImport,
  learnData,
  setLearnData
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [epochs, setEpochs] = useState(100);
  const [lr, setLr] = useState(0.1);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const handleImport = async () => {
    setIsLoading(true);
    try {
      await onImport(epochs, lr);
    } finally {
      setIsLoading(false);
      setLearnData(null);
    }
  };

  const isImportEnabled = wavFile && midFile && !isLoading;

  return (
    <Dialog open={open} onClose={() => !isLoading && onClose()}>
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

        {/* 学習パラメータ設定 */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle1">Settings</Typography>
            <IconButton onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}>
              {showAdvancedSettings ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
          <Collapse in={showAdvancedSettings}>
            <TextField
              label="Epochs"
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              fullWidth
              sx={{ mb: 2 }}
              inputProps={{ min: 1 }}
            />
            <TextField
              label="Learning Rate"
              type="number"
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              fullWidth
              inputProps={{
                min: 0.0001,
                max: 1,
                step: 0.0001
              }}
            />
          </Collapse>
        </Box>

        {learnData && (
          <>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Epoch: {learnData.current_epoch}/{learnData.total_epochs}, Loss: {learnData.loss}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={(learnData.current_epoch / learnData.total_epochs) * 100}
              sx={{ mb: 2 }}
            />
          </>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button onClick={onClose} disabled={isLoading}>Cancel</Button>
          <Button
            variant="contained"
            color="primary"
            sx={{ ml: 2 }}
            onClick={handleImport}
            disabled={!isImportEnabled}
          >
            {isLoading ? (
              <CircularProgress size={16} color="inherit" />
            ) : (
              'Import'
            )}
          </Button>
        </Box>
      </Box>
    </Dialog>
  );
};
