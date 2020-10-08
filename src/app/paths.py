from pathlib import Path

project_root: Path = Path(__file__).absolute().parents[2]

db_name: str = 'DisasterResponse.db'
table_name: str = db_name[:-3]
db_path: Path = project_root / 'src' / 'data' / db_name

model_name: str = 'classifier.pkl'
model_path: Path = project_root / 'src' / 'models' / model_name
