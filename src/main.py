from pipeline.train import ModelTraining
from pipeline.data_ingestion import DataIngestion

if __name__=="__main__":
    
    preprocess = DataIngestion('data/hotel_reservations.csv')
    preprocess.run()

    trainer = ModelTraining('data/train.csv','data/test.csv','model/lgbm_model.pkl')
    trainer.run()