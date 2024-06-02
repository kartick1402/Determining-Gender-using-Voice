import wave

def is_blank_audio(filepath, threshold=0.001):
  try:
    # Open the audio file for reading
    with wave.open(filepath, 'rb') as wav_file:
      # Get frames and frame rate
      frames = wav_file.getnframes()
      frame_rate = wav_file.getframerate()

      # Calculate total squared amplitude
      total_energy = 0
      for frame in wav_file.readframes(1):  # Read one frame at a time
        # Use frame directly (assuming signed short)
        amplitude = frame
        total_energy += amplitude**2

      # Calculate average energy per frame
      average_energy = total_energy / frames

      # Check if average energy is below the threshold
      return average_energy < threshold

  except FileNotFoundError:
    print("File not found")
    return False
  except wave.Error as e:
    print("Error reading WAV file:", e)
    return False

# Example usage
filepath = "blank_audio_test.wav"  # Replace with your file path
threshold = 0.001  # Adjust threshold as needed

if is_blank_audio(filepath, threshold):
  print("The audio appears to be a blank voice note.")
else:
  print("The audio contains sound.")
