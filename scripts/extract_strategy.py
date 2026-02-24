import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine.strategy_extractor import StrategyExtractor

def main():
    print("Initiating Strategy Extraction ML Pipeline...")
    extractor = StrategyExtractor()
    extractor.run_pipeline()

if __name__ == "__main__":
    main()
