from flask import Flask, render_template, request
from Feature_ext import generate_dataset
from rf_model import load_model
from rf_model import retrain_model
from db import db, LogEntry
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///phishing_detection.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

@app.before_request
def create_tables():
    db.create_all()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/getURL', methods=['POST'])
def get_url():
    if request.method == 'POST':
        url = request.form['url']
        print(f"Received URL: {url}")  # Debugging line

        try:
            # Extract features
            data, rank = generate_dataset(url)
            
            # Predict using the model
            rfmodel = load_model('rfmodel.pkl')
            prediction = rfmodel.predict([data])[0]
            prediction_text = "Phishing Website" if prediction == -1 else "Not a Phishing Website"

            if prediction == -1:
                is_phishing = True
                value = "Phishing Website"
            else:
                is_phishing = False
                value = f"{url} is not a Phishing Website!!"

            # Log the result
            new_log = LogEntry(url=url, prediction=value, is_phishing=is_phishing)
            db.session.add(new_log)
            db.session.commit()

            # # Log result to the database
            # log_entry = LogEntry(url=url, prediction=prediction_text)
            # db.session.add(log_entry)
            # db.session.commit()

            return render_template("home.html", error=prediction_text)
        except SQLAlchemyError as e:
            db.session.rollback()
            return render_template("error.html", error=f"Database Error: {str(e)}"), 500
        except Exception as e:
            return render_template("error.html", error=f"Error: {str(e)}"), 500
        
@app.route('/retrain', methods=['POST'])
# Model re-training feature (Not on the frontend)
def retrain():
    api_key = request.headers.get('API-KEY')
    if api_key != os.getenv('ADMIN_API_KEY'):
        return {"error": "Unauthorized"}, 403
    
    try:
        result = retrain_model()
        return {"message": result}, 200
    except Exception as e:
        return {"error": f"Failed to retrain model: {str(e)}"}, 500


@app.route('/manage')
def manage():
    try:
        logs = LogEntry.query.all()
        log_list = [{"id": log.id, "url": log.url, "prediction": log.prediction} for log in logs]
        return {"logs": log_list}, 200
    except SQLAlchemyError as e:
        return {"error": f"Database Error: {str(e)}"}, 500

@app.errorhandler(404)
def not_found_error(e):
    return render_template("error.html", error="Page Not Found"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", error="Internal Server Error"), 500

if __name__ == "__main__":
    app.run(debug=True)
