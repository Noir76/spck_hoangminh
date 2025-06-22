# Fruit Classification App

A Streamlit web application that uses deep learning to classify fruits and vegetables from images. The app can classify 109 different types of fruits, vegetables, and nuts.

## Features

- **Image Upload**: Upload images of fruits/vegetables for classification
- **Webcam Capture**: Take photos directly using your device's camera
- **Real-time Classification**: Get instant predictions with confidence scores
- **User-friendly Interface**: Clean and intuitive Streamlit interface

## Supported Classes

The app can classify 109 different types of fruits, vegetables, and nuts including:

- **Apples**: Apple, Apple Braeburn, Apple Crimson Snow, Apple Delicious, Apple Golden, Apple Granny Smith, Apple Pink Lady
- **Citrus Fruits**: Lemon, Lemon Meyer, Limes, Clementine, Mandarine, Grapefruit, Grapefruit Pink, Tangelo
- **Berries**: Strawberry, Blueberry, Raspberry, Huckleberry, Gooseberry, Mulberry
- **Tropical Fruits**: Mango, Pineapple, Papaya, Guava, Lychee, Rambutan, Passion Fruit
- **Vegetables**: Carrot, Potato, Onion, Cabbage, Cauliflower, Cucumber, Tomato, Eggplant
- **Nuts**: Walnut, Hazelnut, Chestnut, Pistachio, Pecan
- And many more!

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fruit-classification-app.git
cd fruit-classification-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app locally:
```bash
streamlit run app.py
```

## Usage

1. **Upload Image**: Click "Upload Image" and select an image file (JPG, JPEG, PNG)
2. **Use Webcam**: Click "Use Webcam" to take a photo with your camera
3. **Get Results**: Click "Classify" to see the prediction and confidence score

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 100x100 pixels
- **Classes**: 109 different fruits, vegetables, and nuts
- **Framework**: TensorFlow/Keras

## Deployment

This app is designed to be deployed on Streamlit Cloud:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy with the following settings:
   - **Main file path**: `app.py`
   - **Python version**: 3.9+

## File Structure

```
fruit-classification-app/
├── app.py              # Main Streamlit application
├── model_epoch_02.keras # Trained model file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## Requirements

- Python 3.8+
- Streamlit 1.28.0+
- TensorFlow 2.13.0+
- Pillow 10.0.0+
- NumPy 1.24.0+

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 