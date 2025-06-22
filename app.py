from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing import image

# Fruit labels mapping
fruit_labels = {
    0: 'Apple', 1: 'Apple Braeburn', 2: 'Apple Crimson Snow', 3: 'Apple Delicious', 4: 'Apple Golden',
    5: 'Apple Granny Smith', 6: 'Apple Pink Lady', 7: 'Apricot', 8: 'Avocado', 9: 'Avocado ripe',
    10: 'Banana', 11: 'Banana Lady Finger', 12: 'Beans', 13: 'Beetroot', 14: 'Blueberry',
    15: 'Cabbage', 16: 'Cactus fruit', 17: 'Caju seed', 18: 'Cantaloupe', 19: 'Carambula',
    20: 'Carrot', 21: 'Cauliflower', 22: 'Cherimoya', 23: 'Cherry', 24: 'Cherry Rainier',
    25: 'Cherry Sour', 26: 'Cherry Wax', 27: 'Chestnut', 28: 'Clementine', 29: 'Cocos',
    30: 'Corn', 31: 'Corn Husk', 32: 'Cucumber', 33: 'Cucumber Ripe', 34: 'Dates',
    35: 'Eggplant', 36: 'Fig', 37: 'Ginger Root', 38: 'Gooseberry', 39: 'Granadilla',
    40: 'Grape', 41: 'Grape Blue', 42: 'Grape Pink', 43: 'Grapefruit', 44: 'Grapefruit Pink',
    45: 'Guava', 46: 'Hazelnut', 47: 'Huckleberry', 48: 'Kaki', 49: 'Kiwi',
    50: 'Kohlrabi', 51: 'Kumquats', 52: 'Lemon', 53: 'Lemon Meyer', 54: 'Limes',
    55: 'Lychee', 56: 'Mandarine', 57: 'Mango', 58: 'Mangostan', 59: 'Maracuja',
    60: 'Melon Piel de Sapo', 61: 'Mulberry', 62: 'Nectarine', 63: 'Nectarine Flat', 64: 'Nut',
    65: 'Nut Forest', 66: 'Nut Pecan', 67: 'Onion', 68: 'Onion Peeled', 69: 'Papaya',
    70: 'Passion Fruit', 71: 'Peach', 72: 'Peach Flat', 73: 'Pear', 74: 'Pear Abate',
    75: 'Pear Forelle', 76: 'Pear Kaiser', 77: 'Pear Monster', 78: 'Pear Stone', 79: 'Pear Williams',
    80: 'Pepino', 81: 'Pepper', 82: 'Physalis', 83: 'Physalis with Husk', 84: 'Pineapple',
    85: 'Pineapple Mini', 86: 'Pistachio', 87: 'Pitahaya', 88: 'Plum', 89: 'Pomegranate',
    90: 'Pomelo ie', 91: 'Potato', 92: 'Quince', 93: 'Rambutan', 94: 'Raspberry',
    95: 'Salak', 96: 'Strawberry', 97: 'Tamarillo', 98: 'Tangelo', 99: 'Tomato',
    100: 'Tomato Cherry', 101: 'Tomato Heart', 102: 'Walnut', 103: 'Watermelon', 104: 'Zucchini',
    105: 'Berrie', 106: 'Berrie half rippen', 107: 'Berrie not rippen', 108: 'Currant'
}

def preprocess_PIL(pil_img, input_size=(100, 100)):
    pil_img = pil_img.convert("RGB")
    img = pil_img.resize(input_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    test_datagen = image.ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    img_generator = test_datagen.flow(img_array, batch_size=1)
    return img_generator

def main():
    @st.cache_resource
    def load_fruit_model(model_path='model_epoch_02.keras'):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_fruit_model('model_epoch_02.keras')
    st.title("Fruit Classification App")

    option = st.selectbox("Choose input type", ("Upload Image", "Use Webcam"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            if st.button("Classify"):
                img_gen = preprocess_PIL(img)
                predictions = model.predict(next(img_gen))
                prediction_idx = np.argmax(predictions)
                predicted_fruit = fruit_labels.get(prediction_idx, f"Unknown Class {prediction_idx}")
                confidence = np.max(predictions)
                st.success(f"Prediction: **{predicted_fruit}** with {confidence*100:.2f}% confidence.")

    elif option == "Use Webcam":
        img_data = st.camera_input("Take a picture")
        if img_data is not None:
            img = Image.open(img_data)
            st.image(img, caption="Captured Image", use_column_width=True)
            if st.button("Classify Webcam Image"):
                img_gen = preprocess_PIL(img)
                predictions = model.predict(next(img_gen))
                prediction_idx = np.argmax(predictions)
                predicted_fruit = fruit_labels.get(prediction_idx, f"Unknown Class {prediction_idx}")
                confidence = np.max(predictions)
                st.success(f"Prediction: **{predicted_fruit}** with {confidence*100:.2f}% confidence.")

if __name__ == "__main__":
    main()
