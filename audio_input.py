import queue
import sounddevice as sd

# Sampling rate of the audio in samples per second.
# 16,000 Hz means we record 16,000 audio samples every second.
SAMPLE_RATE = 16000

# How many samples we process at once.
# 256 samples at 16,000 Hz ≈ 256 / 16000 ≈ 0.016 seconds per block (~16 ms).
BLOCK_SIZE = 256

# This queue is a thread‑safe buffer between the audio callback
# (which runs in a separate thread) and the main Nengo code.
# The callback puts audio blocks into the queue, and Nengo reads them out.
audio_q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """
    This function is called automatically by sounddevice
    every time a new audio block is available from the microphones.

    Parameters
    ----------
    indata : np.ndarray
        Recorded audio data with shape (frames, channels).
    frames : int
        Number of samples in this block (should match BLOCK_SIZE).
    time : CData
        Timing information from PortAudio (not used here).
    status : CallbackFlags
        Contains error or overflow warnings.
    """
    if status:
        # Print any warning or error (e.g. buffer overflow).
        print(status)

    # Store a copy of the audio block in the queue so the rest of the program
    # can read it later without interfering with the audio thread.
    audio_q.put(indata.copy())


def start_stream(device, channels):
    """
    Start a live audio input stream and return the stream object.

    Parameters
    ----------
    device : int
        Index of the audio device to use (e.g. 5 for your setup).
    channels : int
        Number of input channels (2 for stereo: Mac + iPhone).

    Returns
    -------
    stream : sounddevice.InputStream
        The running audio stream. You can stop() and close() it when finished.
    """
    stream = sd.InputStream(
        device=device,          # which physical or virtual sound device to use
        channels=channels,      # number of microphone channels
        samplerate=SAMPLE_RATE, # how fast we sample
        blocksize=BLOCK_SIZE,   # how many samples per callback
        callback=audio_callback # function that receives audio blocks
    )
    stream.start()
    return stream
