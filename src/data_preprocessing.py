import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Handle missing values, remove duplicates, and drop unnecessary columns."""
    df.drop(columns=['Booking_ID'], inplace=True, errors='ignore')  # Drop ID column
    df.drop_duplicates(inplace=True)  # Remove duplicates
    df.fillna(0, inplace=True)  # Fill missing values (if any) with 0
    return df

def encode_features(df):
    """Convert categorical features into numeric format."""
    label_encoders = {}
    categorical_columns = ['type_of_meal_plan', 'required_car_parking_space', 
                           'room_type_reserved', 'market_segment_type', 
                           'repeated_guest', 'booking_status']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders if needed later
    return df, label_encoders

def preprocess_and_save(input_filepath, output_filepath):
    """Full preprocessing pipeline."""
    df = load_data(input_filepath)
    df = clean_data(df)
    df, _ = encode_features(df)
    df.to_csv(output_filepath, index=False)
    print(f"Processed data saved to {output_filepath}")

if __name__ == "__main__":
    preprocess_and_save("../data/raw_data.csv", "../data/processed_data.csv")
