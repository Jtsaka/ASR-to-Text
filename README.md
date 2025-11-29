# ASR-to-Text

### Brief Overview:

This project aims to implement **Automatic Speech Recognition** (ASR) using the **Open AI's Faster Whisper** model. To capture audio input from a user, the Python library **PyAudio** was utilized. The library allows flexibility in adjusting settings for the desired recordings. The audio file is then saved as a .wav file via the Python wave library.

### Use Case:

Ensure that all necessary dependencies are installed or configured in the working environment (see requirements.txt for details). When the script is run, it prompts the user to "Start Recording Audio." When audio is detected, the state will change to "Voice detected, recording." From there, PyAudio will keep recording any audio input from the user until 3 seconds of silence is detected ("Stop recording"). Upon 3 seconds of silence, the audio is processed and stored as a .wav file as "AudioRec_.wav" with incremental numbering for the file name. If not created, an _AudioRecordings_ directory is created to store the audio recordings for easier management.

When PyAudio detects further audio input following "Stop recording," it will start recording once more, and stop after silence. To optimize this process, **multi-threading** was utilized so that if the user wants to record more audio after a file has already been processed, the transcription process will run in the background via a daemon thread, while allowing the recording function to occur. Transcription processes are placed in a queue for the worker thread to process and transcribe. 

If the user wishes to terminate the process, they can say "stop processing" or hit _Ctrl + c_ to terminate the process.

### Visual Samples:

![Directories_View](/images/AudioTranscriptionDirectories.jpg)

_^^^ Figure 1. Screen shot of _AudioRecordings_ directory created and storing audio files_

![Test_Run](/images/AudioTranscriptionExample.jpg)

_^^^ Figure 2. Screen shot of successful audio transcription process run_

### Resources:

- PyAudio documentation: https://pypi.org/project/PyAudio/
- Faster Whisper documentation: https://github.com/SYSTRAN/faster-whisper
- For the sample wav. file found in "InitialTesting" for samplespeech.wav: https://www.kaggle.com/datasets/pavanelisetty/sample-audio-files-for-speech-recognition?resource=download
