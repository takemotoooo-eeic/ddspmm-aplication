import type { Note } from '../orval/models/backend-api';

export const MIN_NOTE_DURATION_SEC = 0.05;
export const MAX_NOTE_DURATION_SEC = 10;

export const notesEqual = (a: Note, b: Note): boolean =>
  a.start === b.start && a.duration === b.duration && a.frequency === b.frequency;

export const removeNote = (notes: Note[], target: Note): Note[] => {
  const index = notes.findIndex(n => notesEqual(n, target));
  if (index < 0) return notes;
  return [...notes.slice(0, index), ...notes.slice(index + 1)];
};

const noteEnd = (note: Note): number => note.start + note.duration;

/**
 * 単旋律向け: 新規ノート区間と重なる既存ノートをトリム／分割し、新規ノートを追加する。
 */
export const applyMonophonicInsert = (existingNotes: Note[], newNote: Note): Note[] => {
  const newStart = newNote.start;
  const newEnd = newNote.start + newNote.duration;
  const result: Note[] = [];

  for (const note of existingNotes) {
    const s = note.start;
    const e = noteEnd(note);

    if (e <= newStart || s >= newEnd) {
      result.push(note);
      continue;
    }

    if (s < newStart) {
      const leftDuration = newStart - s;
      if (leftDuration >= MIN_NOTE_DURATION_SEC) {
        result.push({ ...note, start: s, duration: leftDuration });
      }
    }
    if (e > newEnd) {
      const rightDuration = e - newEnd;
      if (rightDuration >= MIN_NOTE_DURATION_SEC) {
        result.push({ ...note, start: newEnd, duration: rightDuration });
      }
    }
  }

  result.push(newNote);
  return result.sort((a, b) => a.start - b.start);
};

export const applyMonophonicMove = (
  notes: Note[],
  targetIndex: number,
  movedNote: Note,
): Note[] => {
  if (targetIndex < 0 || targetIndex >= notes.length) return notes;
  const baseNotes = [
    ...notes.slice(0, targetIndex),
    ...notes.slice(targetIndex + 1),
  ];
  return applyMonophonicInsert(baseNotes, movedNote);
};

export const buildNoteFromDrag = (
  anchorTime: number,
  currentTime: number,
  frequency: number,
): { start: number; duration: number; frequency: number } => {
  const clampedCurrentTime =
    currentTime >= anchorTime
      ? Math.min(currentTime, anchorTime + MAX_NOTE_DURATION_SEC)
      : Math.max(currentTime, anchorTime - MAX_NOTE_DURATION_SEC);
  const start = Math.min(anchorTime, clampedCurrentTime);
  const end = Math.max(anchorTime, clampedCurrentTime);
  return { start, duration: end - start, frequency };
};
