from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('house_price_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract input values
            features = [
                float(request.form['MedInc']),
                float(request.form['HouseAge']),
                float(request.form['AveRooms']),
                float(request.form['AveBedrms']),
                float(request.form['Population']),
                float(request.form['AveOccup']),
                float(request.form['Latitude']),
                float(request.form['Longitude'])
            ]

            # Convert input features to numpy array and reshape
            input_features = np.array(features).reshape(1, -1)

            # Predict the price
            predicted_price = model.predict(input_features)[0]
            prediction = f"${predicted_price * 1000:.2f}"  # Formatting the prediction
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
