import os
import pyaudio
import wave
from array import array
from faster_whisper import WhisperModel
import threading
import queue

shutdown_event = threading.Event() #event flag
end_condition = "stop processing"

RECORDINGS_DIRECTORY = "AudioRecordings"
os.makedirs(RECORDINGS_DIRECTORY, exist_ok = True) #create if the directory doesn't exist yet

CHUNK = 1024 #size in bytes
FORMAT = pyaudio.paInt16 #usually 16 or 24 bits (higher quality) -- bit depth
CHANNELS = 1 #1 = mono (less data to process), 2 = stereo 
RATE = 44100 #sampling freq in hertz, generally 44100 or 48000 hertz
MIN_VOLUME = 500 #min interger sample value
SILENCE_LEN = 3 #in sec
MODEL_SIZE = "small.en"

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def record_instance(output_file):
    """
    Records audio -- takes in the filename to save audio file
    
    Args:
        output_file: name of audio file

    Returns: none
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True, #use device as input (i.e. mic), specified bc can have output option
                    frames_per_buffer=CHUNK)
    
    print("Start recording audio")
    frames = []
    voice_detected = False
    silent_chunks_counter = 0

    try:

        while True:
            if shutdown_event.is_set():
                print("Aborting recording process")
                break
            data = stream.read(CHUNK) #stream reads 1024 frames -- returns raw bytes
            audio_data = array('h', data) #interprets bytes as 16-bit signed bytes #
            volume = max(abs(sample) for sample in audio_data)

            if not voice_detected:
                if volume >= MIN_VOLUME:
                    print("Voice detected, recording. ")
                    voice_detected = True
                    frames.append(data)
                    silent_chunks_counter = 0
                else: #no voice detected
                    continue

            else:
                frames.append(data) #if already recording
                if volume < MIN_VOLUME:
                    silent_chunks_counter += 1
                    if(silent_chunks_counter * CHUNK / RATE) >= SILENCE_LEN:
                        print("Stop recording")
                        break

                else:
                    silent_chunks_counter = 0
    finally: #Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    if not frames: #no audio captures
        print("No audio was captured or detected")
        return

    ###Saving audio as a wave file
    wf = wave.open(output_file, 'wb') #set to write binary
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT)) #gets number of bytes per sample
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames)) #b"" is empty bytes object used as separator -- when concatenated gives one long byte object containing audio frames back to back -- can't pass in list by itself
    wf.close()
    print(f"File saved as {output_file}")

def transcribe(audio_path):
    """
    Takes in the filename to transcribe, then processes the audio recording. Transcription checks to see whether "stop processing" flag is detected from user input to end program.
    
    Args:
        audio_path: name of audio file

    Returns: none
    """
    segments, _ = model.transcribe(audio_path, beam_size=5) #beam size refers to beam search, how many best partial transcriptions to analyze before deciding on single best
    text = "".join(seg.text for seg in segments)
    print(f"{text}") #print text transcription

    if end_condition in text.lower():
        shutdown_event.set()

    return text

def transcribe_background(audio_queue):
    """
    Worker that gets audio file from queue and transcribes in the background
    
    Args:
        audio_queue: queue with audio file names
    
    Returns: none
    """
    while True:
        path = audio_queue.get() #waits until item in queue
        if path is None: #terminates if end of process
            break
        try:
            transcribe(path)
        finally:
            audio_queue.task_done()

def run_process():
    """
    Main function that creates queue and worker and starts transcription process until terminated. All audio files are stored in the "AudioRecordings" directory for neatness.
    
    Args: none
    
    Returns: none
    """
    audio_queue = queue.Queue()
    worker = threading.Thread(target=transcribe_background,
                              args=(audio_queue,), #expects tuple of args, provide 1 tuple
                              daemon=True) #daemon thread to run in the background without blocking main program
    worker.start()
    audio_file_index = 1

    try:
        while not shutdown_event.is_set(): #only breaks when "Ctrl + c"
            filename = f"AudioRec{audio_file_index}.wav"
            full_path = os.path.join(RECORDINGS_DIRECTORY, filename) 
            record_instance(full_path)
            if shutdown_event.is_set():
                break

            audio_queue.put(full_path)
            audio_file_index += 1
            
    except KeyboardInterrupt:
        print("Keyboard interrupt, stopping process")
    
    finally:
        audio_queue.put(None)
        worker.join() #waits for worker thread to finish, then exits cleanly
        print("Terminating process.")


if __name__ == "__main__":
    run_process()
    