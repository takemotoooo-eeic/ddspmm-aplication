import { Box, Button, Dialog, Typography } from '@mui/material';
import {
  DiffusionSettingsEditor,
} from '../editors/DiffusionSettingsEditor';

interface DiffusionSettingsDialogProps {
  open: boolean;
  onClose: () => void;
  numDenoisingSteps: number;
  onNumDenoisingStepsChange: (value: number) => void;
}

export const DiffusionSettingsDialog = ({
  open,
  onClose,
  numDenoisingSteps,
  onNumDenoisingStepsChange,
}: DiffusionSettingsDialogProps) => {
  return (
    <Dialog open={open} onClose={onClose}>
      <Box sx={{ bgcolor: 'background.paper', p: 3, minWidth: 480 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          DIFFUSION SETTINGS
        </Typography>
        <DiffusionSettingsEditor
          numDenoisingSteps={numDenoisingSteps}
          onNumDenoisingStepsChange={onNumDenoisingStepsChange}
        />
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button variant="contained" onClick={onClose}>
            Close
          </Button>
        </Box>
      </Box>
    </Dialog>
  );
};
