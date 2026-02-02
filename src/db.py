from pymongo import MongoClient
from datetime import datetime


class MongoLogger:
    def __init__(self, uri="mongodb://localhost:27017", db="anpr"):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db]
        self.logs = self.db.vehicle_logs
        self.videos = self.db.video_logs
        
        # Test connection on init
        try:
            self.client.admin.command('ping')
            print("✅ MongoDB connected successfully")
        except Exception as e:
            print(f"⚠️ MongoDB connection warning: {e}")

    def save_vehicle(self, vehicle_type, plate, confidence):
        """Synchronous vehicle logging to MongoDB"""
        try:
            self.logs.insert_one({
                "timestamp": datetime.utcnow(),
                "vehicle_type": vehicle_type,
                "plate_number": plate,
                "confidence": round(float(confidence), 2)
            })
            return True
        except Exception as e:
            print(f"❌ Failed to save vehicle: {e}")
            return False

    def save_video(self, path, reason):
        """Synchronous video logging to MongoDB"""
        try:
            self.videos.insert_one({
                "video_path": path,
                "reason": reason,
                "timestamp": datetime.utcnow()
            })
            return True
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
            return False
