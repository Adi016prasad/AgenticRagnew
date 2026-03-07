import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

class fireBaseServices:
    def __init__(self):
        cred = credentials.Certificate(PATH_OF_CERTIFICATEOFFIREBASE_DATABASE)
        self.firebase_app = firebase_admin.initialize_app(cred)
        self.firebase_database = firestore.client()