import librosa
import numpy as np
from keras.models import load_model

# Load the saved model
loaded_model = load_model('urbanVibe.keras')

# Function to preprocess audio data
def preprocess_audio(audio_file):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    
    # Extract features (you can adjust this based on how your model was trained)
    # For example, let's extract Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)  # Adjust n_mfcc to match the expected input shape
    
    # Pad or truncate MFCCs to match the expected input shape (40, 167)
    if mfccs.shape[1] < 167:
        mfccs = np.pad(mfccs, ((0, 0), (0, 167 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :167]
    
    # Normalize the features
    mfccs_normalized = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    # Expand dimensions to match model input shape
    mfccs_normalized = np.expand_dims(mfccs_normalized, axis=-1)
    
    return mfccs_normalized

# Path to your audio file
audio_file_path = r'C:\Users\zekiy\OneDrive\Masaüstü\archive (2)\fold10\2937-1-0-0.wav'

# Preprocess the audio data
preprocessed_audio = preprocess_audio(audio_file_path)

# Make predictions
predictions = loaded_model.predict(preprocessed_audio)

# Assuming your classes are ['class1', 'class2', ...]
# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Map the class index to the class label
classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# Debugging statements
print("Predicted class index:", predicted_class_index)
print("Number of classes:", len(classes))

# Check if predicted_class_index is within the range of the classes list
if 0 <= predicted_class_index < len(classes):
    predicted_class = classes[predicted_class_index]
    print("Predicted class:", predicted_class)
else:
    print("Error: Predicted class index is out of range.")