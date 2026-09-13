import { useState, useCallback } from 'react';

export type FlightMode = 'studio' | 'fpv';

export function useFlightMode(initialMode: FlightMode = 'studio') {
  const [flightMode, setFlightMode] = useState<FlightMode>(initialMode);

  const enterFpv = useCallback(() => setFlightMode('fpv'), []);
  const exitFpv = useCallback(() => setFlightMode('studio'), []);
  const toggleFlightMode = useCallback(() => {
    setFlightMode((prev) => (prev === 'studio' ? 'fpv' : 'studio'));
  }, []);

  return {
    flightMode,
    isFpv: flightMode === 'fpv',
    isStudio: flightMode === 'studio',
    setFlightMode,
    enterFpv,
    exitFpv,
    toggleFlightMode,
  };
}
