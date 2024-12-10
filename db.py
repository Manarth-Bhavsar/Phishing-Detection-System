from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class LogEntry(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    url = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    is_phishing = db.Column(db.Boolean, nullable=False)  # True for phishing, False otherwise

    def __repr__(self):
        return f"<LogEntry {self.id}: {self.url} - {self.prediction} - {self.is_phishing}>"
