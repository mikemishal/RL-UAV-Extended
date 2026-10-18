import { useSyncExternalStore } from 'react';

export interface PlaybackClockSnapshot {
  replayId: string | null;
  timeS: number;
}

let snapshot: PlaybackClockSnapshot = { replayId: null, timeS: 0 };
const listeners = new Set<() => void>();

export function setPlaybackClock(replayId: string | null, timeS: number): void {
  if (snapshot.replayId === replayId && snapshot.timeS === timeS) return;
  snapshot = { replayId, timeS };
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot(): PlaybackClockSnapshot {
  return snapshot;
}

export function usePlaybackClock(replayId: string | null): number | null {
  const current = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return current.replayId === replayId && replayId !== null ? current.timeS : null;
}
