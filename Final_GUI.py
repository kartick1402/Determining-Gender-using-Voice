import os
import tkinter as tk
from tkinter import filedialog as fd
import tkinter.messagebox
import tkinterDnD
import customtkinter as cs
import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
import speech_recognition as sr
r = sr.Recognizer()
from test import is_blank_audio,has_word_hi,is_silent,normalize,trim,add_silence,record,record_to_file,extract_feature

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

cs.set_ctk_parent_class(tkinterDnD.Tk)
cs.set_appearance_mode("Dark")
cs.set_default_color_theme("blue")

root=cs.CTk()
root.geometry("1000x500")
root.title("Gender Detector Using Voice")

from utils import load_data, split_data, create_model
model = create_model()
model.load_weights("results/model.h5")

def openFile():
    f_types1=[('WAV files','*.wav')]
    openFile.filepath = fd.askopenfilename(filetypes=f_types1)
    label_1.configure(text=(openFile.filepath))
    file = openFile.filepath
    if not os.path.exists(file) or not file.endswith(".wav"):
            print("The filepath you entered doesn't exist or is not of .wav format. Please Try Again...")
            quit()
    else:
        if is_blank_audio(file,0.001):
            print("The audio appears to be a blank voice note.")
        else:
            print("The audio contains sound.")
        if has_word_hi(file):
            print("The audio contains the word 'hi'. Again record the voice. Quitting....")
            quit()
        else:
            print("The audio does not contain the word 'hi'.")
        duration = librosa.get_duration(path=file)
        if duration < 30:
            print("The voice note should be longer than 30 seconds. Please upload or record a longer voice note.")
            quit()    
    # extract features and reshape it
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    # show the result!
    print("Result:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")

def starting_voice_analysing():
    file = "test.wav"
    record_to_file(file)
    if is_blank_audio(file,0.001):
        print("The audio appears to be a blank voice note.")
    else:
        print("The audio contains sound.")
    if has_word_hi(file):
        print("The audio contains the word 'hi'. Again record the voice. Quitting....")
        quit()
    else:
        print("The audio does not contain the word 'hi'.")
    duration = librosa.get_duration(path=file)
    if duration < 30:
        print("The voice note should be longer than 30 seconds. Please upload or record a longer voice note.")
        quit()
    # extract features and reshape it
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    # show the result!
    print("Result:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")


frame=cs.CTkFrame(master=root)
frame.pack(pady=20,padx=60, fill="both", expand=True)

label = cs.CTkLabel(master=frame, text="Gender Identification Using Voice", justify=cs.LEFT)
label.pack(pady=10, padx=10)

tabview_1 = cs.CTkTabview(master=frame, width=400, height=250,)
tabview_1.pack(pady=10, padx=10)

tab1=tabview_1.add("CHOOSE ACTION")
button1=cs.CTkButton(master=tab1, text="Upload Audio from Device", command=openFile)
button1.pack(pady=12, padx=10)
label_1 = cs.CTkLabel(master=tab1,text="",justify=cs.LEFT)
label_1.pack(pady=10, padx=10)

button2=cs.CTkButton(master=tab1, text="Record Your Voice", command=starting_voice_analysing)
button2.pack(pady=12, padx=10)
label_2 = cs.CTkLabel(master=tab1,text="",justify=cs.LEFT)
label_2.pack(pady=10, padx=10)

root.mainloop()