import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

#settings for faster-whisper
samplerate = 16000 #number of audio samples taken per second -- more = accurate = slower bc more processing
block_duration = 0.5 #size of audio chunk that sd library has to process at a time -- smaller block duration = more responsive
chunk_duration = 2 #in seconds (time of audio collected before sending to transcriber)
channels = 1 #set to record one audio/one mic

frames_per_block = int(samplerate * block_duration) #from secs to audio frames
frames_per_chunk = int(samplerate* chunk_duration) #from secs to audio frames

audio_queue = queue.Queue() #thread-safe data structure that acts as comm channel. Recorder thread puts newly recorded audio blocks into queue, to which the transcriber thread gets them from the queue (prevents race conditions and ensures data integrity)

audio_buffer = [] #simple list that temp stores audio blocks recieved by transcriber until enough data to form a chunk_duration

#Model setup
model = WhisperModel("small.en", device="cpu", compute_type="int8")

def audio_callback(indata, frames, time, status): #called automatically by sounddevice whenever new block of audio is ready (takes audio data indata and puts it into audio_queue) - nonblocking, doesn't slow down recording process
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder(): #sets up audio input stream and is run as own thread. Continuously records audio from mic, callbackfunction handles data transfer to audio_queue
    with sd.InputStream(samplerate=samplerate, channels=channels,callback=audio_callback, blocksize=frames_per_block):
        print("listening... ctrl+c to stop")
        while True:
            sd.sleep(100)

def transcriber(): #run with own thread, continuously checks audio_queue for new data. When it has enough data to make full chunk, concatenates blocks, passes audio through Whisper model, then prints result
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = [] #clear buffer

            audio_data = audio_data.flatten().astype(np.float32)

            #transcription without timestamp
            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=1 #Max speed
            )

            for segment in segments:
                print(f"{segment.text}") #just print text, no timestamps

#start threads
threading.Thread(target=recorder, daemon=True).start()
threading.Thread(target=transcriber, daemon=True).start()

try:
    while True:
        sd.sleep(100) # Sleep for a short duration to save CPU cycles
except KeyboardInterrupt:
    print("\nExiting program...")