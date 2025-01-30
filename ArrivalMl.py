from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


connect_str = "DefaultEndpointsProtocol=https;AccountName=mlpw5g48966911887;AccountKey=mrAhppNTLycLK1odEtWiGfubfle5TJfUMwex5QV2+Q8Ggk6kMSvBUXwY+JTILISiJjRqNhEIr/oJ+AStpDybJQ==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)


container_name = "flights"
blob_name = "flights_weather.csv"

# Crea il blob client utilizzando il blob_service_client che presumibilmente hai già definito.
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# Scarica il blob e leggilo direttamente in un DataFrame pandas
data = io.BytesIO(blob_client.download_blob().readall())
df = pd.read_csv(data, low_memory=False)

# Remove rows where any of these columns have null values
df.dropna(subset=['DepTime', 'ArrTime', 'DepDelay', 'ArrDelay', 'rain (mm)Dep', 'rain (mm)Arr'], inplace=True)

# Remove diverted flights
df = df[df['Diverted'] == 0]

# Define rain categories
def categorize_rain(amount):
    if amount == 0:
        return 'No rain'
    elif amount <= 5:
        return 'Light rain'
    elif amount <= 15:
        return 'Moderate rain'
    else:
        return 'Heavy rain'

# Apply the categorization to departure and arrival rain data
df['rain_category_Dep'] = df['rain (mm)Dep'].apply(categorize_rain)
df['rain_category_Arr'] = df['rain (mm)Arr'].apply(categorize_rain)

# Optionally, drop the original rain columns if no longer needed
df.drop(['rain (mm)Dep', 'rain (mm)Arr'], axis=1, inplace=True)

# List of columns to keep for the arrival model
arrival_columns = [
    'ArrDelay', 'ArrDel15', 'Dest', 'CRSArrTime', 'TaxiIn',
    'rain_category_Arr', 'wind_speed_10m (km/h)Arr',
    'wind_direction_10m (°)Arr', 'temperature_2m (°C)Arr', 'relative_humidity_2m (%)Arr',
    'cloud_cover (%)Arr', 'weather_code (wmo code)Arr'
]

# Reduce the dataframe to only the arrival-related columns
df_arrivals = df[arrival_columns]

# List of provided airport codes
airport_codes = [
    "PLN", "BTV", "CWA", "TWF", "ABE", "MLI", "PHF", "OME", "VLD", "ROA", "PSM", "JAN", "DCA", "MDT", "HOB",
    "SCC", "BNA", "ACK", "SHR", "MKE", "LGA", "RDM", "IDA", "OAJ", "MYR", "PSG", "JAX", "MSN", "JMS", "SHD",
    "AUS", "TOL", "CHA", "AMA", "MGM", "ABY", "SAT", "ISP", "MEM", "LBF", "HPN", "ONT", "HYS", "COS", "BGR",
    "BDL", "PAE", "MFR", "SCK", "STX", "ABR", "FLL", "SAN", "FAI", "CSG", "DAL", "ALO", "RAP", "DHN", "MTJ",
    "ECP", "HHH", "ELP", "LGB", "INL", "BRD", "CGI", "AVL", "DEC", "SPS", "RFD", "HGR", "BOI", "PIH", "LWS",
    "SUX", "BRW", "DSM", "BPT", "WYS", "IAD", "MAF", "HOU", "BFL", "OWB", "TUL", "CLT", "ADQ", "PAH", "CNY",
    "HRL", "DTW", "ICT", "ESC", "JLN", "GPT", "TYR", "LBE", "KTN", "GFK", "FWA", "SLC", "OTH", "LIH", "BLI",
    "JNU", "SMX", "HIB", "EAR", "BQN", "MVY", "FAR", "XNA", "BWI", "ORF", "ACV", "USA", "RIW", "LSE", "RSW",
    "ROW", "IAH", "HTS", "FNT", "ALS", "LAX", "PSC", "SRQ", "EGE", "FSD", "GEG", "MKG", "PIE", "BMI", "GST",
    "TLH", "PWM", "BGM", "ATW", "GRR", "BIS", "XWA", "SJC", "LYH", "BUF", "BZN", "AGS", "SEA", "AVP", "ADK",
    "DFW", "PSE", "STT", "SBN", "DEN", "PUW", "LNK", "LWB", "LRD", "PBI", "TXK", "ABI", "FLG", "OKC", "SMF",
    "SUN", "EKO", "MLB", "MBS", "BRO", "CHS", "LAN", "AZA", "TPA", "RST", "MQT", "JAC", "ELM", "SYR", "LBL",
    "DDC", "RNO", "MEI", "ACT", "EYW", "CYS", "SBA", "PVD", "CRP", "IAG", "CDC", "PIA", "SIT", "PBG", "DAY",
    "CIU", "PUB", "YAK", "SJT", "LFT", "APN", "SAV", "BKG", "ITO", "EWR", "ILG", "SNA", "GRK", "GSO", "MSY",
    "LIT", "GSP", "ACY", "HYA", "PSP", "GTR", "DLG", "DBQ", "SGF", "FSM", "GUC", "IND", "TVC", "MRY", "YKM",
    "LAR", "SJU", "PNS", "SWF", "BET", "CLL", "COD", "OGG", "CRW", "DAB", "GRI", "KOA", "CMX", "TYS", "DVL",
    "RHI", "SDF", "MSO", "GRB", "SBP", "SWO", "GJT", "OTZ", "ANC", "HDN", "MIA", "MOT", "BTR", "TBN", "OMA",
    "DIK", "EVV", "PIT", "EWN", "SGU", "EAU", "MHK", "EUG", "YUM", "LBB", "DRO", "MLU", "DRT", "MCI", "PGD",
    "FCA", "GCC", "GNV", "BLV", "HLN", "FAY", "BUR", "MOB", "FOD", "BQK", "CHO", "ASE", "RIC", "COU", "DLH",
    "MSP", "VEL", "ROC", "BHM", "ORD", "VPS", "LAW", "GTF", "CPR", "STC", "CMH", "CMI", "SHV", "ORH", "MHT",
    "PDX", "SLN", "RDU", "CVG", "BJI", "RKS", "CDV", "BIL", "CLE", "ATL", "LEX", "STL", "HSV", "SCE", "RDD",
    "GCK", "PHX", "WRG", "AZO", "EAT", "BOS", "ERI", "SFO", "BTM", "CAK", "LAS", "PIB", "MDW", "OGS", "PVU",
    "CAE", "AKN", "ALB", "PHL", "LCH", "TEX", "SAF", "ABQ", "AEX", "ALW", "BFF", "BIH", "CID", "CKB", "FAT",
    "GGG", "HNL", "ILM", "IMT", "ITH", "JFK", "JST", "LCK", "MCO", "MCW", "MFE", "OAK", "PRC", "SFB", "SPI",
    "STS", "TRI", "TTN", "TUS", "TWS", "VCT"
]

# Creating an encoding map sorted alphabetically
airport_labels = {code: i for i, code in enumerate(sorted(airport_codes))}

# Adding a new column to the DataFrame with the encoded values
df_arrivals['Dest_encoded'] = df_arrivals['Dest'].map(airport_labels)


# Creating an instance of LabelEncoder
encoder = LabelEncoder()

# Fitting the encoder and transforming the 'rain_category_Arr' column
df_arrivals['rain_category_Arr_encoded'] = encoder.fit_transform(df_arrivals['rain_category_Arr'])

# Optionally, to see the mapping from categorical labels to integers
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))



# Define features and target
X = df_arrivals.drop('ArrDelay', axis=1)  # all other columns are features
y = df_arrivals['ArrDelay']  # ArrDelay is the target

# Split the dataset into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Convert 'CRSArrTime' from 'HH:MM' format to minutes past midnight
df_arrivals['CRSArrTime'] = df_arrivals['CRSArrTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

# Define features and target variable, ensuring no non-numeric columns are included
X = df_arrivals.drop(['ArrDelay', 'Dest', 'rain_category_Arr'], axis=1)  # Adjust this line if there are other non-numeric columns
y = df_arrivals['ArrDelay']

y = y.apply(
    lambda row:
        'early' if row < 0
        else ('0 min - 1 hour' if row <= 60
        else ('1-3 hours' if row <= 180
        else 'more than 3 hours'
        ))
)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Continue with the setup of the RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predicting and evaluating
y_pred_class = classifier.predict(X_test)


# Printing classification report
report = classification_report(y_test, y_pred_class, digits=4)