import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load the ResNet50 model with pretrained ImageNet weights
model = ResNet50(weights='imagenet')

# Function to predict animal species
def predict_animal_species(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Return the predictions
    return decoded_predictions

# Streamlit app interface
def main():
    st.title("Animal Species Prediction using ResNet50")
    
    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Predict the animal species
        predictions = predict_animal_species(image)

        # Display the predictions
        for _, animal, probability in predictions:
            st.write(f"{animal}: {probability:.2%}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
