import pandas as pd    
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler



class WeatherDataset(Dataset):
    def __init__(self,csv_path,input_window=168,output_window=72):
        self.input_window = input_window
        self.output_window = output_window
        
        
        # 📥 لود داده‌ها
        df = pd.read_csv(csv_path)
        df = df[["time","temperature"]].copy()
        
        # 🧼 حذف NaN و ریست ایندکس
        df.dropna(inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        
        # 🔢 نرمال‌سازی دما
        self.scaler = MinMaxScaler()
        df["temperature"] = self.scaler.fit_transform(df[["temperature"]])
        
        
        
        self.temps = df["temperature"].values.astype(np.float32)
        
        
        self.X , self.y = self.create_sequences()
        
        
        
        
    def create_sequences(self):
        X,y = [], []
        total_len = len(self.temps)
        
        for i in range(total_len - self.input_window - self.output_window):
            x_seq = self.temps[i:i+self.input_window]
            y_seq = self.temps[i+self.input_window:i+self.input_window + self.output_window]
            X.append(x_seq)
            y.append(y_seq)
            
        return np.array(X),np.array(y)
    
    
    
    
    
    def __len__(self):
        return len(self.X) 
    
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx] 
    
    
    
    
    def get_scaler(self):
        return self.scaler  
        
        
        
        
