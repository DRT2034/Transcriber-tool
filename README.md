# Audio transcription with OpenAI's Whisper model

A small Python script that transcribes an audio or video file with OpenAI Whisper and exports the transcript as a readable PDF (which can easily be summarized by an LLM to contain a good level of granularity).

## How it works

1. Put an `.mp4`, `.mp3`, `.wav`, or other supported media file in the project folder.
2. Open `transcribe.py`.
3. Change `audio_path` to your file name.
4. Run the script.
5. A PDF transcript is generated automatically.

## Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


# when starting: start virtual environment (in terminal) source .venv/bin/activate
# change the name of the file being transcribed and run script
# info on whisper model: https://openai.com/nl-NL/index/whisper/
# When done: type 'deactivate' in terminal