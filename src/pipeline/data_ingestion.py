import os
import boto3
import pandas as pd
from dotenv import load_dotenv
from zlogger.logger import ZLogger
import configparser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.common_function import load_data
# Initialize the config and logger
path_file = "config/logging.ini"
config = configparser.ConfigParser()
config.read(path_file)

logg = ZLogger("hotel_reservation_system", config)

# Load environment variables
logg.info("Loading environment variables...")
load_dotenv()

class DataIngestion:

    def __init__(self, data_path):

        self.data_path = data_path

        # Get AWS credentials from .env
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION = os.getenv("AWS_REGION")
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
        self.S3_FILE_NAME = os.getenv("S3_FILE_NAME")

        # Check if any environment variables are missing
        if not all([self.AWS_ACCESS_KEY_ID, self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, self.S3_BUCKET_NAME, self.S3_FILE_NAME]):
            logg.error("Missing one or more required environment variables.")
            raise ValueError("Missing one or more required environment variables.")

        logg.info("Environment variables loaded successfully.")

        # Set up S3 client
        logg.info(f"Setting up S3 client for region: {self.AWS_REGION}...")
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
                region_name=self.AWS_REGION,
            )
            logg.info("S3 client setup successful.")
        except Exception as e:
            logg.error(f"Error setting up S3 client: {str(e)}")
            raise

        # Download the file from S3
        local_file_path = "data/hotel_reservations.csv"
        logg.info(f"Attempting to download file {self.S3_FILE_NAME} from S3 bucket {self.S3_BUCKET_NAME}...")
        try:
            s3.download_file(self.S3_BUCKET_NAME, self.S3_FILE_NAME, local_file_path)
            logg.info(f"File {self.S3_FILE_NAME} successfully downloaded from S3 bucket {self.S3_BUCKET_NAME}.")
        except Exception as e:
            logg.error(f"Error downloading file from S3: {str(e)}")
            raise


    def clean_data(self, df):
        """Handle missing values, remove duplicates, and drop unnecessary columns."""
        logg.info("Cleaning data...")
        df.drop(columns=['Booking_ID'], inplace=True, errors='ignore')  # Drop ID column
        df.drop_duplicates(inplace=True)  # Remove duplicates
        df.fillna(0, inplace=True)  # Fill missing values (if any) with 0
        return df

    def encode_features(self, df):
        """Convert categorical features into numeric format."""
        logg.info("Encoding categorical features...")
        label_encoders = {}
        categorical_columns = ['type_of_meal_plan', 'required_car_parking_space', 
                            'room_type_reserved', 'market_segment_type', 
                            'repeated_guest', 'booking_status']
    
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Save encoders if needed later
        return df, label_encoders

    def split_data(self, df):
        """Split data into train, test, and validation sets."""
        logg.info("Splitting data into train, test, and validation sets...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, test_df

    def preprocess_and_save(self):
        """Full preprocessing pipeline."""
        df = load_data(self.data_path)
        df = self.clean_data(df)
        df, _ = self.encode_features(df)
        
        # Split the data
        train_df, test_df = self. split_data(df)
        
        # Save to CSV files
        train_df.to_csv('data/train.csv', index=False)
        test_df.to_csv('data/test.csv', index=False)
        
        logg.info("Data split and saved to train.csv, test.csv, and valid.csv.")
    
    def run(self):
        self.preprocess_and_save()

