from firebase_setup import db

# Reference to the 'eye_tracking' node
db_ref = db.reference("eye_tracking/sample_data")

# Your sample data
sample_data = [
    {"date": "2025-03-03", "time": "18:49:08:213", "val": 1},
    {"date": "2025-03-03", "time": "18:49:08:463", "val": 1},
    {"date": "2025-03-03", "time": "18:49:08:713", "val": 0},
    {"date": "2025-03-03", "time": "18:49:08:963", "val": 1},
    {"date": "2025-03-03", "time": "18:49:09:213", "val": 0},
    {"date": "2025-03-03", "time": "18:49:09:463", "val": 1},
    {"date": "2025-03-03", "time": "18:49:09:713", "val": 1},
    {"date": "2025-03-03", "time": "18:49:09:963", "val": 0},
    {"date": "2025-03-03", "time": "18:49:10:213", "val": 0},
    {"date": "2025-03-03", "time": "18:49:10:463", "val": 1},
    {"date": "2025-03-03", "time": "18:49:10:713", "val": 1},
    {"date": "2025-03-03", "time": "18:49:10:963", "val": 1},
    {"date": "2025-03-03", "time": "18:49:11:213", "val": 1},
    {"date": "2025-03-03", "time": "18:49:11:463", "val": 0},
    {"date": "2025-03-03", "time": "18:49:11:713", "val": 0},
]

# Upload each entry
for entry in sample_data:
    new_ref = db_ref.push(entry)
    print(f"✅ Uploaded entry with key: {new_ref.key}")

print("🎉 All data uploaded successfully!")
