import { useRef, useEffect } from 'react';

export interface DroneInputState {
  pitchForward: boolean; // W / ArrowUp
  pitchBackward: boolean; // S / ArrowDown
  yawLeft: boolean; // A / ArrowLeft
  yawRight: boolean; // D / ArrowRight
  throttleUp: boolean; // Space / Q
  throttleDown: boolean; // E
  turbo: boolean; // Shift
}

export interface DroneInputOptions {
  onModeSelect?: (mode: 'manual' | 'orbit' | 'transect' | 'noe') => void;
  onGimbalToggle?: () => void;
  onCameraModeToggle?: () => void;
  onQuickPanelToggle?: () => void;
  onThemeToggle?: () => void;
}

/**
 * useDroneInput:
 * Event-driven keyboard input listener for FPV drone flight mechanics.
 * Maps WASD, Arrow keys, Space, Q, E, Shift, and shortcut keys (1/2/3/4/G/T/P/L).
 */
export function useDroneInput(active: boolean = true, options?: DroneInputOptions) {
  const inputRef = useRef<DroneInputState>({
    pitchForward: false,
    pitchBackward: false,
    yawLeft: false,
    yawRight: false,
    throttleUp: false,
    throttleDown: false,
    turbo: false,
  });

  useEffect(() => {
    if (!active) {
      inputRef.current = {
        pitchForward: false,
        pitchBackward: false,
        yawLeft: false,
        yawRight: false,
        throttleUp: false,
        throttleDown: false,
        turbo: false,
      };
      return;
    }

    const onKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === 'w' || key === 'arrowup') inputRef.current.pitchForward = true;
      if (key === 's' || key === 'arrowdown') inputRef.current.pitchBackward = true;
      if (key === 'a' || key === 'arrowleft') inputRef.current.yawLeft = true;
      if (key === 'd' || key === 'arrowright') inputRef.current.yawRight = true;
      if (key === ' ' || key === 'q') inputRef.current.throttleUp = true;
      if (key === 'e') inputRef.current.throttleDown = true;
      if (key === 'shift') inputRef.current.turbo = true;

      // Autopilot & utility hotkey triggers (§4.3)
      if (key === '1') options?.onModeSelect?.('manual');
      if (key === '2') options?.onModeSelect?.('orbit');
      if (key === '3') options?.onModeSelect?.('transect');
      if (key === '4') options?.onModeSelect?.('noe');
      if (key === 'g') options?.onGimbalToggle?.();
      if (key === 't') options?.onCameraModeToggle?.();
      if (key === 'p') options?.onQuickPanelToggle?.();
      if (key === 'l') options?.onThemeToggle?.();
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === 'w' || key === 'arrowup') inputRef.current.pitchForward = false;
      if (key === 's' || key === 'arrowdown') inputRef.current.pitchBackward = false;
      if (key === 'a' || key === 'arrowleft') inputRef.current.yawLeft = false;
      if (key === 'd' || key === 'arrowright') inputRef.current.yawRight = false;
      if (key === ' ' || key === 'q') inputRef.current.throttleUp = false;
      if (key === 'e') inputRef.current.throttleDown = false;
      if (key === 'shift') inputRef.current.turbo = false;
    };

    const onBlur = () => {
      inputRef.current = {
        pitchForward: false,
        pitchBackward: false,
        yawLeft: false,
        yawRight: false,
        throttleUp: false,
        throttleDown: false,
        turbo: false,
      };
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, [active, options]);

  return inputRef;
}

export default useDroneInput;
