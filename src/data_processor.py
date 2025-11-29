import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\Eva\Desktop\Equipo-equipo-debug-queens-5\data\dataset.csv")
files, columnes=df.shape
print (f"Nombre de files{files}")