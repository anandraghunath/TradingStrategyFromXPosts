import os
import pandas as pd
from typing import Dict, Any, List

class TradeStorage:
    def __init__(self, data_dir: str = "data", filename: str = "trades_dataset.csv"):
        self.data_dir = data_dir
        self.filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def save_trades(self, trades: List[Dict[str, Any]]):
        """
        Appends a list of trade dictionaries to the CSV dataset.
        If the file doesn't exist, it creates it with headers.
        """
        if not trades:
            return

        df = pd.DataFrame(trades)
        
        # If file exists, append without writing headers again.
        if os.path.exists(self.filepath):
            df.to_csv(self.filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(self.filepath, mode='w', header=True, index=False)
            
        print(f"Successfully saved {len(trades)} trades to {self.filepath}")

    def load_trades(self) -> pd.DataFrame:
        """
        Loads the historical dataset into a Pandas DataFrame for ML processing.
        """
        if os.path.exists(self.filepath):
            return pd.read_csv(self.filepath)
        else:
            print(f"Dataset {self.filepath} does not exist yet.")
            return pd.DataFrame()
