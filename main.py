from vosk import Model, KaldiRecognizer
import pyaudio
import json

# Load model
model = Model(r"vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Initialize mic
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000,
                  input=True, input_device_index=1, frames_per_buffer=8192)
stream.start_stream()

print("Listening... (Press Ctrl+C to stop)")

try:
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print(text)
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    mic.terminate()