from pathlib import Path
from processors.processor import ImageProcessor

DIR_IN = Path(__file__).resolve().parent / 'images' / 'raw'
DIR_OUT = Path(__file__).resolve().parent / 'images' / 'processed'

processor = ImageProcessor(DIR_IN, DIR_OUT, (300, 300))
processor.execute()
