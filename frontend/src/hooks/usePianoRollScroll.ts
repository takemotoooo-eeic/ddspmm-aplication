import { useCallback, useRef } from 'react';

export function usePianoRollScroll() {
  const timelineRef = useRef<HTMLDivElement>(null);
  const pianoRollRef = useRef<HTMLDivElement>(null);
  const keysRef = useRef<HTMLDivElement>(null);
  const scrollLeftRef = useRef(0);
  const scrollTopRef = useRef(0);
  const syncingVerticalRef = useRef(false);

  const syncScrollLeft = useCallback((scrollLeft: number) => {
    scrollLeftRef.current = scrollLeft;
    if (timelineRef.current) timelineRef.current.scrollLeft = scrollLeft;
    if (pianoRollRef.current) pianoRollRef.current.scrollLeft = scrollLeft;
  }, []);

  const syncScrollTop = useCallback(
    (scrollTop: number, source: 'keys' | 'pianoRoll') => {
      if (syncingVerticalRef.current) return;
      syncingVerticalRef.current = true;
      scrollTopRef.current = scrollTop;
      if (source !== 'keys' && keysRef.current) {
        keysRef.current.scrollTop = scrollTop;
      }
      if (source !== 'pianoRoll' && pianoRollRef.current) {
        pianoRollRef.current.scrollTop = scrollTop;
      }
      syncingVerticalRef.current = false;
    },
    [],
  );

  const handlePianoRollScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      const { scrollLeft, scrollTop } = event.currentTarget;
      syncScrollLeft(scrollLeft);
      syncScrollTop(scrollTop, 'pianoRoll');
    },
    [syncScrollLeft, syncScrollTop],
  );

  const handleKeysScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      syncScrollTop(event.currentTarget.scrollTop, 'keys');
    },
    [syncScrollTop],
  );

  const syncScrollTopTo = useCallback((scrollTop: number) => {
    if (syncingVerticalRef.current) return;
    syncingVerticalRef.current = true;
    scrollTopRef.current = scrollTop;
    if (keysRef.current) keysRef.current.scrollTop = scrollTop;
    if (pianoRollRef.current) pianoRollRef.current.scrollTop = scrollTop;
    syncingVerticalRef.current = false;
  }, []);

  return {
    timelineRef,
    pianoRollRef,
    keysRef,
    scrollLeftRef,
    scrollTopRef,
    syncScrollLeft,
    syncScrollTopTo,
    handlePianoRollScroll,
    handleKeysScroll,
  };
}
