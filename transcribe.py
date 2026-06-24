# when starting: start virtual environment (in terminal) source .venv/bin/activate
# change the name of the file being transcribed and run script
# info on whipser model: https://openai.com/nl-NL/index/whisper/

# When done: type 'deactivate' in terminal
#example full session: cd /Users/asterckx/Documents/Transcriber-tool
#source .venv/bin/activate
#python transcribe.py
#deactivate
#for whisper-cpp (no mp4 unless convert see gpt): use command ./build/bin/whisper-cli -m models/ggml-medium.en.bin -f ../escape.mp3 -otxt



import os
import torch
import whisper
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  
from reportlab.lib.styles import getSampleStyleSheet  
import re 
from deepmultilingualpunctuation import PunctuationModel



# Force CPU for now, as GPU is problematic with MAC 
device = "cpu"
print("Using device:", device)

model = whisper.load_model("medium", device=device) #put it on medium for now, for very large might use small

audio_path = "DataGen.mp4"
result = model.transcribe(audio_path, language="en", verbose=True)

print("\n--- TRANSCRIPT ---\n")
print(result["text"][:500])  # preview first 500 characters

# ---- PDF Export with Paragraphs ----
base_name = os.path.splitext(audio_path)[0]  
pdf_path = f"{base_name}.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter) 
styles = getSampleStyleSheet()  
story = []  

# Split text into sentences, then regroup into paragraphs
sentences = re.split(r'(?<=[.!?]) +', result["text"])  
paragraph = "" 
for i, sentence in enumerate(sentences, 1):  
    paragraph += sentence + " "  
    if i % 4 == 0:  # every 4 sentences = new paragraph  
        story.append(Paragraph(paragraph.strip(), styles["Normal"]))  
        story.append(Spacer(1, 12)) 
        paragraph = ""  

# add leftover text if any
if paragraph:  
    story.append(Paragraph(paragraph.strip(), styles["Normal"])) 

doc.build(story)  
print(f"Transcript saved to {pdf_path}")
