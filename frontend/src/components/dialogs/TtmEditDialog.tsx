import {
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogContent,
  TextField,
  Typography,
} from '@mui/material';
import { useState } from 'react';
import { TTM_MAX_SEGMENT_SEC } from '../../constants/editor';

interface TtmEditDialogProps {
  open: boolean;
  startSec: number;
  endSec: number;
  anchor: { top: number; left: number } | null;
  loading: boolean;
  onClose: () => void;
  onSubmit: (text: string) => Promise<void>;
}

export const TtmEditDialog = ({
  open,
  startSec,
  endSec,
  anchor,
  loading,
  onClose,
  onSubmit,
}: TtmEditDialogProps) => {
  const [text, setText] = useState('');
  const segmentLen = endSec - startSec;
  const tooLong = segmentLen > TTM_MAX_SEGMENT_SEC;

  const handleSubmit = async () => {
    if (!text.trim() || tooLong) return;
    await onSubmit(text.trim());
    setText('');
  };

  const handleClose = () => {
    if (loading) return;
    setText('');
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      hideBackdrop
      disableEnforceFocus
      PaperProps={{
        sx: {
          position: 'fixed',
          top: anchor ? Math.min(anchor.top, window.innerHeight - 220) : '40%',
          left: anchor ? Math.min(anchor.left, window.innerWidth - 340) : '50%',
          transform: anchor ? undefined : 'translate(-50%, -50%)',
          m: 0,
          width: 320,
          bgcolor: 'background.paper',
          border: '1px solid #444',
        },
      }}
    >
      <DialogContent sx={{ p: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Edit region ({startSec.toFixed(2)}s – {endSec.toFixed(2)}s)
        </Typography>
        {tooLong && (
          <Typography variant="caption" color="error" sx={{ display: 'block', mb: 1 }}>
            Maximum region length is {TTM_MAX_SEGMENT_SEC} seconds
          </Typography>
        )}
        <TextField
          fullWidth
          size="small"
          multiline
          minRows={2}
          placeholder="e.g. energetic jazz piano with drums"
          value={text}
          onChange={e => setText(e.target.value)}
          disabled={loading}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void handleSubmit();
            }
          }}
        />
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 2 }}>
          <Button size="small" onClick={handleClose} disabled={loading}>
            Cancel
          </Button>
          <Button
            size="small"
            variant="contained"
            onClick={() => void handleSubmit()}
            disabled={!text.trim() || tooLong || loading}
          >
            {loading ? <CircularProgress size={16} color="inherit" /> : 'Submit'}
          </Button>
        </Box>
      </DialogContent>
    </Dialog>
  );
};
