import firebase_admin
from firebase_admin import credentials, db

# Load Firebase credentials
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://eyehealthmonitor-default-rtdb.firebaseio.com/"
})

# Reference to the database
db_ref = db.reference("eye_tracking")

print("🔥 Firebase is set up and connected!")
