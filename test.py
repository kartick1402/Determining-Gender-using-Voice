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
import time

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_blank_audio(filepath, threshold=0.001):
  try:
    with wave.open(filepath, 'rb') as wav_file:
      frames = wav_file.getnframes()
      frame_rate = wav_file.getframerate()

      # Finding the total squared amplitude
      total_energy = 0
      for frame in wav_file.readframes(1):  # Reading one frame at a time for better evaluation
        amplitude = frame
        total_energy += amplitude**2

      average_energy = total_energy / frames

      return average_energy < threshold

  except FileNotFoundError:
    print("File not found")
    return False
  except wave.Error as e:
    print("Error reading WAV file:", e)
    return False

def has_word_hi(filepath):
  r = sr.Recognizer()
  try:
    with sr.AudioFile(filepath) as source:
      audio_data = r.record(source)

    text = r.recognize_google(audio_data).lower() 
    return "hi" in text
  
  except sr.UnknownValueError:
    print("Could not understand audio")
    return False
  except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return False
  
def is_silent(snd_data):
    #Returns 'True' if below the 'silent' threshold
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    #Making the volume average
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Removing manually the blank spots at the start and end of audio for better understanding"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    # Adding silence to the start and end of the data
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')
    while 1:
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    print("Please Start Speaking....")
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def extract_feature(file_name, **kwargs):
    #Extracting numerous feature from audio file given as argument
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

from speech_recognition import AudioData
if __name__ == "__main__":

    from utils import load_data, split_data, create_model

    model = create_model()
    model.load_weights("results/model.h5")

    cmd=input("CHOOSE U FOR UPLOAD AND R FOR RECORD:")
    if cmd.lower()=="u":
        file=input("Enter FILEPATH:")
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
    elif cmd.lower()=="r":
        print("Please talk")
        # put the file name here
        file = "test.wav"
        # record the file (start talking)
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
    else:
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