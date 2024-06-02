import speech_recognition as sr

def has_word_hi(filepath):
  r = sr.Recognizer()
  try:
    with sr.AudioFile(filepath) as source:
      audio_data = r.record(source)

    text = r.recognize_google(audio_data).lower()
    print(text) 
    return "hi" in text
  
  except sr.UnknownValueError:
    print("Could not understand audio")
    return False
  except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return False

# Example usage
filepath = "FINAL_COMPLETE_AUDIO.wav" 

if has_word_hi(filepath):
  print("The audio contains the word 'hi'.")
else:
  print("The audio does not contain the word 'hi'.")