import time
import numpy as np
try:
    import sounddevice as sd
except (ImportError, OSError) as exc:      # no PortAudio library (e.g. a container): beeps are off
    sd = None
    print(f"[play_beep] sound is not available, beeps are off: {exc}")
_AUDIO_OK = sd is not None                 # switched off after the first failed playback

# Wake up the sound card with 0.1 seconds of silence
#sd.play(np.zeros(4410*10), 44100)
#sd.wait()


def generate_wave(shape, frequency, t):
    """Generates a raw waveform based on the requested shape."""
    shape = shape.lower()
    if shape == "sine":
        return np.sin(2 * np.pi * frequency * t)
    elif shape == "square":
        return np.sign(np.sin(2 * np.pi * frequency * t))
    elif shape in ("saw", "sawtooth"):
        return 2.0 * (frequency * t - np.floor(0.5 + frequency * t))
    else:
        raise ValueError(f"Unsupported shape '{shape}'. Choose 'sine', 'square', or 'saw'.")

def play_beep(
    frequency=440.0,
    duration=0.5,
    volume=0.2,
    shape="sine",
    repeats=1,
    pause=0.0,
    sample_rate=44100
):
    """Plays a sound with specified wave properties. Without a usable sound output it does
    nothing: a beep must never stop teleoperation (it is called from the control loop)."""
    global _AUDIO_OK
    if not _AUDIO_OK:
        return
    volume = max(0.0, min(1.0, volume))
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = volume * generate_wave(shape, frequency, t)

    try:
        for i in range(repeats):
            sd.play(wave, sample_rate)
            sd.wait()
            if i < repeats - 1 and pause > 0:
                time.sleep(pause)
    except Exception as exc:               # e.g. sounddevice.PortAudioError: no output device
        _AUDIO_OK = False
        print(f"[play_beep] no usable sound output, beeps are off: {exc}")

def play_arpegio(
    duration=0.1, 
    volume=0.2, 
    shape="square", 
    gap=0.03, 
    arpeggio = [261.63, 329.63, 392.00]
):
    """Plays a C Major arpeggio."""
    for freq in arpeggio:
        play_beep(
            frequency=freq,
            duration=duration,
            volume=volume,
            shape=shape,
            repeats=1
        )
        if gap > 0:
            time.sleep(gap)

# Test it out directly
if __name__ == "__main__":
    play_beep(volume=0.001, duration=0.5)
    play_arpegio(arpeggio = [261.63, 329.63, 392.00])
    play_arpegio(arpeggio = [392.00, 329.63, 261.63])
    