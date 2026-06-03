import { TTM_MAX_SEGMENT_SEC } from '../constants/editor';
import { pixelToSec } from './waveformSelection';

/** 波形要素上の X [px]（0 … waveformWidth） */
export const waveformXFromClient = (
  clientX: number,
  container: HTMLElement,
  waveformWidth: number,
): number => {
  const rect = container.getBoundingClientRect();
  return Math.max(0, Math.min(waveformWidth, clientX - rect.left));
};

export const maxTtmSpanPx = (waveformWidth: number, durationSec: number): number => {
  if (durationSec <= 0 || waveformWidth <= 0) return waveformWidth;
  return (TTM_MAX_SEGMENT_SEC / durationSec) * waveformWidth;
};

/** 選択幅が TTM 最大区間長（30秒）に達しているか（anchor = ドラッグ開始位置） */
export const isAtMaxTtmSpan = (
  anchorStartPx: number,
  endPx: number,
  waveformWidth: number,
  durationSec: number,
  epsilonPx = 0.5,
): boolean => {
  const maxSpanPx = maxTtmSpanPx(waveformWidth, durationSec);
  return Math.abs(endPx - anchorStartPx) >= maxSpanPx - epsilonPx;
};

/**
 * TTM 区間選択を最大 30 秒に収める（開始位置は固定し、終端のみクランプ）。
 */
export const clampTtmSelectionPx = (
  anchorStartPx: number,
  endPx: number,
  waveformWidth: number,
  durationSec: number,
): { startPx: number; endPx: number } => {
  const maxSpanPx = maxTtmSpanPx(waveformWidth, durationSec);
  if (endPx >= anchorStartPx) {
    if (endPx - anchorStartPx > maxSpanPx) {
      return { startPx: anchorStartPx, endPx: anchorStartPx + maxSpanPx };
    }
    return { startPx: anchorStartPx, endPx };
  }
  if (anchorStartPx - endPx > maxSpanPx) {
    return { startPx: anchorStartPx, endPx: anchorStartPx - maxSpanPx };
  }
  return { startPx: anchorStartPx, endPx };
};

/**
 * 30秒上限時に、選択をさらに広げる方向への端スクロールだけを抑止する。
 * - 右方向ドラッグ: 右端スクロールを止める
 * - 左方向ドラッグ: 左端スクロールを止める
 */
export const shouldEdgeScroll = (
  anchorStartPx: number,
  endPx: number,
  clientX: number,
  scrollContainer: HTMLElement,
  marginPx: number,
  waveformWidth: number,
  durationSec: number,
): boolean => {
  const { endPx: clampedEnd } = clampTtmSelectionPx(
    anchorStartPx,
    endPx,
    waveformWidth,
    durationSec,
  );
  if (!isAtMaxTtmSpan(anchorStartPx, clampedEnd, waveformWidth, durationSec)) {
    return true;
  }

  const view = scrollContainer.getBoundingClientRect();
  const atLeftEdge = clientX < view.left + marginPx;
  const atRightEdge = clientX > view.right - marginPx;

  if (clampedEnd >= anchorStartPx) {
    return !atRightEdge;
  }
  return !atLeftEdge;
};

/** @deprecated clampTtmSelectionPx を使用 */
export const clampSelectionEndPx = (
  startPx: number,
  endPx: number,
  waveformWidth: number,
  durationSec: number,
): number => clampTtmSelectionPx(startPx, endPx, waveformWidth, durationSec).endPx;

/**
 * ドラッグ中にビューポート端へマウスがあるとき横スクロールする。
 * 端に近いほど速くスクロールする。
 */
export const applyEdgeScroll = (
  scrollContainer: HTMLElement,
  clientX: number,
  marginPx: number,
  maxSpeedPx: number,
): boolean => {
  const view = scrollContainer.getBoundingClientRect();
  let scrolled = false;

  if (clientX < view.left + marginPx) {
    const dist = Math.max(0, marginPx - (clientX - view.left));
    const speed = Math.max(2, (dist / marginPx) * maxSpeedPx);
    const prev = scrollContainer.scrollLeft;
    scrollContainer.scrollLeft = Math.max(0, prev - speed);
    scrolled = scrollContainer.scrollLeft !== prev;
  } else if (clientX > view.right - marginPx) {
    const dist = Math.max(0, clientX - (view.right - marginPx));
    const speed = Math.max(2, (dist / marginPx) * maxSpeedPx);
    const maxScroll = scrollContainer.scrollWidth - scrollContainer.clientWidth;
    const prev = scrollContainer.scrollLeft;
    scrollContainer.scrollLeft = Math.min(maxScroll, prev + speed);
    scrolled = scrollContainer.scrollLeft !== prev;
  }

  return scrolled;
};

export { pixelToSec };
