/** 波形上のピクセル位置を秒に変換 */
export const pixelToSec = (px: number, width: number, durationSec: number): number => {
  if (width <= 0 || durationSec <= 0) return 0;
  return Math.max(0, Math.min(durationSec, (px / width) * durationSec));
};

export const normalizeSelection = (
  startPx: number,
  endPx: number,
  width: number,
  durationSec: number,
): { startSec: number; endSec: number } => {
  const a = pixelToSec(Math.min(startPx, endPx), width, durationSec);
  const b = pixelToSec(Math.max(startPx, endPx), width, durationSec);
  const minLen = 0.05;
  if (b - a < minLen) {
    return { startSec: a, endSec: Math.min(durationSec, a + minLen) };
  }
  return { startSec: a, endSec: b };
};
