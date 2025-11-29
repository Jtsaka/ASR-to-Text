from faster_whisper import WhisperModel

model_size = "small.en"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("sampleSpeech.wav", beam_size=5) #can remove "info", but should specific language as pparameter inside paretheses as (language="en")

# Segments var stores chunks of transcribed audio. Each segment contains information about specifc portion of audio (i.e. segment.start, .end, .text (transcribed text))

# Info var stores meta data about transcription process (i.e. detected lang, lang probability)

# Beam_size is a parameter controlling number of possible transcriptions to track at each step of process. Larger size = more possibilites = more accurate, but more computationally intensive + more time, opposite for smaller size (less possibilities less accurate, but less computationally intensive + faster). 5 is good default

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))