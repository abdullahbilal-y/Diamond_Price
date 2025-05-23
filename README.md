# DiamPrice - Diamond Price Estimator

A machine learning web application that predicts diamond prices based on diamond characteristics.

## Overview

DiamPrice is a Streamlit-powered web application that uses a trained machine learning model to estimate diamond prices based on their physical characteristics. Users can input features like carat weight, cut quality, color grade, clarity, and dimensions to receive an instant price prediction.

## Features

- **Instant Price Prediction**: Get real-time diamond price estimates
- **User-friendly Interface**: Simple form for entering diamond characteristics
- **Multiple Diamond Attributes**: Considers carat, cut, color, clarity, depth, table, and dimensions
- **Prediction History**: Saves all predictions for future reference

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diamprice.git
   cd diamprice
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter the diamond's characteristics in the form:
   - Carat (weight)
   - Cut (quality of the cut)
   - Color (diamond color grade)
   - Clarity (how clear the diamond is)
   - Depth (total depth percentage)
   - Table (width of top of diamond relative to widest point)
   - x, y, z dimensions (length, width, height in mm)

2. Click the "Predict" button to get the estimated price.

3. The prediction will be displayed and automatically saved to the prediction history.

## Technical Details

- **Model**: Random Forest Regressor trained on diamond pricing data
- **Features**: The model considers 9 key features that impact diamond pricing
- **Performance**: Trained with a random forest algorithm for reliable predictions
- **Preprocessing**: Categorical features (cut, color, clarity) are encoded using label encoders

## Project Structure

```
diamprice/
├── app.py                  # Streamlit web application
├── model_training.ipynb    # Jupyter notebook for model training
├── diamond_model.joblib    # Trained machine learning model
├── label_encoders.joblib   # Saved label encoders for categorical features
├── prediction_history.csv  # CSV file storing prediction history
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Future Improvements

- Add data visualization for diamond characteristics
- Implement model comparison and evaluation metrics
- Create a feature to explain the prediction results
- Add ability to upload bulk diamond data for batch predictions

## License

[MIT License](LICENSE)

## Contact

For questions or suggestions, please open an issue on the GitHub repository.
