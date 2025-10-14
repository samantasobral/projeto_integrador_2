import pandas as pd
import pickle
import numpy as np
import json
from geopy.distance import geodesic

class PredictDelivery(object):
    def __init__(self):
        self.Delivery_person_Age_scaler = pickle.load(open('../parameter/Delivery_person_Age_scaler.pkl', 'rb'))
        self.Delivery_person_Ratings_scaler = pickle.load(open('../parameter/Delivery_person_Ratings_scaler.pkl', 'rb'))
        self.Weatherconditions_label_encoder = pickle.load(open('../parameter/Weatherconditions_label_encoder.pkl', 'rb'))
        self.Vehicle_condition_scaler = pickle.load(open('../parameter/Vehicle_condition_scaler.pkl', 'rb'))
        self.multiple_deliveries_scaler = pickle.load(open('../parameter/multiple_deliveries_scaler.pkl', 'rb'))
        self.distance_scaler = pickle.load(open('../parameter/distance_scaler.pkl', 'rb'))
    
    def data_formatation(self, df):
        df['Weatherconditions'] = df['Weatherconditions'].str.strip()
        return df
    
    def calculate_time_diff(self,df):
        time_diff = df['Time_Order_picked'] - df['Time_Orderd']
        time_diff = time_diff.apply(lambda x: x + pd.Timedelta(days=1) if x.days < 0 else x)
        df['order_prepare_time'] = time_diff.dt.total_seconds() / 60
        df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace = True)
        return df
    
    def calculate_distance(self, df):
        df['distance'] = np.zeros(len(df))
        restaurante_coordinates = df[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
        cliente_coordinates = df[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()
        df['distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in zip(restaurante_coordinates, cliente_coordinates)])
        df['distance'] = df['distance'].astype('str').str.extract('(\d+)').astype('int64')
        return df
    
    def feature_engineering(self, df2):
        df2 = self.calculate_time_diff(df2)
        df2 = self.calculate_distance(df2)
        return df2
    
    def data_preparation(self, df3):
        df3['Delivery_person_Age'] = self.Delivery_person_Age_scaler.transform(df3[['Delivery_person_Age']])
        df3['Delivery_person_Ratings'] = self.Delivery_person_Ratings_scaler.transform(df3[['Delivery_person_Ratings']])
        df3['Vehicle_condition'] = self.Vehicle_condition_scaler.transform(df3[['Vehicle_condition']])
        df3['multiple_deliveries'] = self.multiple_deliveries_scaler.transform(df3[['multiple_deliveries']])
        df3['distance'] = self.distance_scaler.transform(df3[['distance']])
        
        ordem_categoria = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}
        df3['Road_traffic_density'] = df3['Road_traffic_density'].map(ordem_categoria)

        df3['Weatherconditions'] = self.Weatherconditions_label_encoder.transform(df3[['Weatherconditions']])

        df3 = df3[['Delivery_person_Age', 'Delivery_person_Ratings', 'Weatherconditions',
                    'Road_traffic_density', 'Vehicle_condition', 'multiple_deliveries', 'distance']]     
        return df3
    
    def get_predictions(self, model, test_data, original_data):
        pred = model.predict(test_data)
        original_data['prediction'] = pred
        return original_data.to_json(orient='records')