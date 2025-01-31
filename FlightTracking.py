import pandas as pd
import streamlit as st
import requests
from datetime import datetime
import pytz
import folium
from folium.plugins import MarkerCluster, AntPath
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.metric_cards import style_metric_cards
from geopy.distance import geodesic
import numpy as np
from geographiclib.geodesic import Geodesic
 
 
 
def aviation_api_call(dep_iata, arr_iata, flight_number=None):
    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": "a20a60cfdf780bf8e0ca59097c387d6a",
        "dep_iata": dep_iata,
        "arr_iata": arr_iata,
    }
   
    if flight_number:
        params["flight_iata"] = flight_number
   
    response = requests.get(url, params=params)
   
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"❌ Error: {response.status_code}, {response.text}")
        return None
 
# Funzione per chiamare l'API WeatherAPI per i dati meteo
def weather_api_call(iata_code, days=1):
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": "b901edd192c548feb88141117252201",  
        "q": f"iata:{iata_code}",  
        "days": days,
        "aqi": "yes",
        "alerts": "yes",  
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else None
 
# Funzione per convertire l'ora UTC in orario locale
def convert_utc_to_local(utc_time_str):
    utc_time = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%S+00:00")
    utc_time = pytz.utc.localize(utc_time)
    return utc_time.strftime("%Y-%m-%d %H:%M:%S")
 
# Funzione per estrarre solo le ore divisibili per 3
def filter_forecast_by_hour(forecast_data):
    hours = []
    temperatures = []
   
    for hour in forecast_data:
        hour_time = hour["time"].split(" ")[1]
        hour_int = int(hour_time.split(":")[0])
       
        if hour_int % 3 == 0:
            hours.append(hour_time)
            temperatures.append(hour["temp_c"])
   
    return hours, temperatures
 
# Funzione per caricare i dati del CSV con gli aeroporti
def load_airport_data_and_coordinates():
    df = pd.read_csv("airports_data_cleaned.csv")
   
    airport_coords = {}
   
    for _, row in df.iterrows():
        airport_coords[row['Origin']] = (row['LatitudeDep'], row['LongitudeDep'])
        airport_coords[row['Dest']] = (row['LatitudeArr'], row['LongitudeArr'])
   
    return df, airport_coords
 
# Carica i dati dell'CSV e le coordinate
airport_data, airport_coords = load_airport_data_and_coordinates()
 
# Configura Streamlit
st.set_page_config(page_title="Flight Tracker", page_icon="🌍", layout="wide")
 
# Stile della pagina
style_metric_cards(border_color="#4CAF50", background_color="#F1F8E9")
st.markdown(
    """<style>
        .main {background-color: #f4f4f9;}
        .stButton > button {background-color: #0078D7; color: white; font-weight: bold; border-radius: 5px;}
        .stMarkdown h1 {color: #FF5722; font-size: 3rem; text-align: center;}
    </style>""",
    unsafe_allow_html=True
)
 
# Intestazione della pagina
st.markdown(
    """<h1>🌌 Flight & Weather Tracker</h1>""",
    unsafe_allow_html=True
)
add_vertical_space(2)
 
# Selezione partenza e arrivo
st.sidebar.header("🛏 Airport Selections")
st.sidebar.write("Select your departure and arrival details below:")
 
# Selezione obbligatoria degli aeroporti
states = sorted(airport_data['OriginStateName'].unique())
departure_state = st.sidebar.selectbox("Departure State", states)
departure_cities = airport_data[airport_data['OriginStateName'] == departure_state]['OriginCityName'].unique()
departure_city = st.sidebar.selectbox("Departure City", departure_cities)
departure_airports = airport_data[airport_data['OriginCityName'] == departure_city]['Origin'].unique()
departure_iata = st.sidebar.selectbox("Departure Airport (IATA Code)", departure_airports)
 
arrival_state = st.sidebar.selectbox("Arrival State", sorted(airport_data['DestStateName'].unique()))
arrival_cities = airport_data[airport_data['DestStateName'] == arrival_state]['DestCityName'].unique()
arrival_city = st.sidebar.selectbox("Arrival City", arrival_cities)
arrival_airports = airport_data[airport_data['DestCityName'] == arrival_city]['Dest'].unique()
arrival_iata = st.sidebar.selectbox("Arrival Airport (IATA Code)", arrival_airports)  
 
# Filtro opzionale per numero di volo
st.sidebar.write("Optional: Search by Flight Number")
flight_number = st.sidebar.text_input("Enter Flight Number (e.g., AA123)", "")
 
 
# Funzione per calcolare la linea geodetica curva
def calculate_geodetic_curve(start_coords, end_coords, num_points=100):
    geod = Geodesic.WGS84
    lats, lons = [], []
 
    # Calcola punti intermedi lungo la rotta geodetica
    line = geod.InverseLine(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
    for i in np.linspace(0, line.s13, num_points):
        point = line.Position(i)
        lats.append(point['lat2'])
        lons.append(point['lon2'])
 
    # Restituisci i punti come lista di tuple
    return list(zip(lats, lons))
 
# Funzione per ottenere la bounding box con margine
def get_bounding_box_with_margin(start_coords, end_coords, margin=0.5):
    min_lat = min(start_coords[0], end_coords[0]) - margin
    max_lat = max(start_coords[0], end_coords[0]) + margin
    min_lon = min(start_coords[1], end_coords[1]) - margin
    max_lon = max(start_coords[1], end_coords[1]) + margin
    return [min_lat, min_lon, max_lat, max_lon]
 
# Pulsante per ottenere informazioni
if st.sidebar.button("Get Flight Info"):
    if departure_iata in airport_coords and arrival_iata in airport_coords:
        departure_coords = airport_coords[departure_iata]
        arrival_coords = airport_coords[arrival_iata]
 
        # Calcolare la bounding box per adattare la mappa alla rotta con margine
        bounds = get_bounding_box_with_margin(departure_coords, arrival_coords)
 
        # Mappa con zoom sulla rotta tra partenza e arrivo
        st.markdown("### 🌍 United States Map")
        m = folium.Map(
            location=[(departure_coords[0] + arrival_coords[0]) / 2, (departure_coords[1] + arrival_coords[1]) / 2],
            zoom_start=6,
            tiles="CartoDB positron"
        )
 
        # Aumentare il numero di punti per ottenere una curva più precisa
        curve_points = calculate_geodetic_curve(departure_coords, arrival_coords, num_points=50)
 
        # Creare la curva con AntPath
        folium.plugins.AntPath(curve_points, weight=6, color="blue", opacity=0.8).add_to(m)
 
        # Aggiungere il marker di partenza
        folium.Marker(
            departure_coords,
            popup=f"<strong>Departure:</strong> {departure_iata}",
            icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
            tooltip="Departure Airport"
        ).add_to(m)
 
        # Aggiungere il marker di arrivo
        folium.Marker(
            arrival_coords,
            popup=f"<strong>Arrival:</strong> {arrival_iata}",
            icon=folium.Icon(color="red", icon="plane", prefix="fa"),
            tooltip="Arrival Airport"
        ).add_to(m)
 
        # Aggiungere il controllo zoom (per il pulsante di reset)
        folium.plugins.Fullscreen(position="topright").add_to(m)
 
        # Adattare la vista alla rotta con margine
        m.fit_bounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])
 
        # Visualizzare la mappa
        folium_static(m, width=800, height=500)
       
 
    def extract_weather_info(weather_data):
        if weather_data:
            location = weather_data["location"]["name"]
            current = weather_data["current"]
            return {
                "City": location,
                "Temperature (°C)": current["temp_c"],
                "Condition": current["condition"]["text"],
                "Humidity (%)": current["humidity"],
                "Wind Speed (km/h)": current["wind_kph"],
                "Feels Like (°C)": current["feelslike_c"],
                "Wind Direction": current["wind_dir"],
                "Pressure (mb)": current["pressure_mb"],
                "Precipitation (mm)": current["precip_mm"],
                "UV Index": current["uv"]
            }
        return None
 
    if departure_iata and arrival_iata:
        st.markdown("---")
        st.markdown("## 🌤️ Weather Information for Departure & Arrival Cities")
        st.markdown("---")
       
        departure_weather = extract_weather_info(weather_api_call(departure_iata))
        arrival_weather = extract_weather_info(weather_api_call(arrival_iata))
       
        if departure_weather and arrival_weather:
            weather_df = pd.DataFrame({
                "Departure City": departure_weather.values(),
                "Arrival City": arrival_weather.values()
            }, index=departure_weather.keys())
           
            st.table(weather_df)
       
        st.markdown("---")
 
        # Previsioni meteo per la partenza
        departure_weather = weather_api_call(departure_iata, days=1)
        if departure_weather:
            forecast_data_departure = departure_weather["forecast"]["forecastday"][0]["hour"]
            hours_departure, temperatures_departure = filter_forecast_by_hour(forecast_data_departure)
 
            # Visualizzazione delle allerte meteo per la partenza
            if "alerts" in departure_weather and isinstance(departure_weather["alerts"], list):
                for alert in departure_weather["alerts"]:
                    st.markdown(f"#### 🚨 Weather Alert for Departure: {departure_weather['location']['name']}")
                    st.write(f"**Alert Type:** {alert['event']}")
                    st.write(f"**Severity:** {alert['severity']}")
                    st.write(f"**Urgency:** {alert['urgency']}")
                    st.write(f"**Description:** {alert['desc']}")
                    st.write(f"**Effective:** {convert_utc_to_local(alert['effective'])}")
                    st.write(f"**Expires:** {convert_utc_to_local(alert['expires'])}")
                    st.markdown("---")
            else:
                st.write(f"No weather alerts available for Departure: {departure_weather['location']['name']}.")
           
        # Previsioni meteo per l'arrivo
        arrival_weather = weather_api_call(arrival_iata, days=1)
        if arrival_weather:
            forecast_data_arrival = arrival_weather["forecast"]["forecastday"][0]["hour"]
            hours_arrival, temperatures_arrival = filter_forecast_by_hour(forecast_data_arrival)
           
   
            # Visualizzazione delle allerte meteo per l'arrivo
            if "alerts" in arrival_weather and isinstance(arrival_weather["alerts"], list):
                for alert in arrival_weather["alerts"]:
                    st.markdown(f"#### 🚨 Weather Alert for Arrival: {arrival_weather['location']['name']}")
                    st.write(f"**Alert Type:** {alert['event']}")
                    st.write(f"**Severity:** {alert['severity']}")
                    st.write(f"**Urgency:** {alert['urgency']}")
                    st.write(f"**Description:** {alert['desc']}")
                    st.write(f"**Effective:** {convert_utc_to_local(alert['effective'])}")
                    st.write(f"**Expires:** {convert_utc_to_local(alert['expires'])}")
                    st.markdown("---")
            else:
                st.write(f"No weather alerts available for Arrival: {arrival_weather['location']['name']}.")
           
            st.markdown("---")
     
    # Grafico meteo
    if departure_weather and arrival_weather:
        fig, ax = plt.subplots(figsize=(8, 3))  # Crea una figura e un asse espliciti
        sns.set_theme(style="whitegrid")
       
        # Traccia la temperatura per la partenza
        ax.plot(hours_departure, temperatures_departure, marker="o", color='#38A1DB', markersize=6, linestyle='-', linewidth=2, label=f"Departure: {departure_weather['location']['name']}")
       
        # Traccia la temperatura per l'arrivo
        ax.plot(hours_arrival, temperatures_arrival, marker="o", color='red', markersize=6, linestyle='-', linewidth=2, label=f"Arrival: {arrival_weather['location']['name']}")
       
        # Titolo e legende
        ax.set_title("Weather Forecast for Departure and Arrival", fontsize=14, fontweight='bold', color='#003366', pad=20)
        ax.set_xlabel("Hour", fontsize=12, fontweight='bold', color='#003366')
        ax.set_ylabel("Temperature (°C)", fontsize=12, fontweight='bold', color='#003366')
        ax.set_xticklabels(hours_departure, rotation=45)
        ax.legend()
       
        # Aggiungere un piccolo offset per evitare sovrapposizioni delle etichette
        offset_departure = -3.0  # Posiziona ulteriormente sotto la linea per la partenza
        offset_arrival = 1.5  # Posiziona sopra la linea per l'arrivo
 
        # Etichette dei valori per la partenza (sotto la linea)
        for i, temp in enumerate(temperatures_departure):
            ax.text(hours_departure[i], temp + offset_departure, f"{temp}°C", ha='center', fontsize=9, color='#444444')
 
        # Etichette dei valori per l'arrivo (sopra la linea)
        for i, temp in enumerate(temperatures_arrival):
            ax.text(hours_arrival[i], temp + offset_arrival, f"{temp}°C", ha='center', fontsize=9, color='#444444')
 
        plt.tight_layout()  # Ottimizzare il layout
        st.pyplot(fig)  # Passa esplicitamente la figura a st.pyplot
       
        st.markdown("---")
 
 
    # Dettagli del volo
    st.markdown("### ✈️ Flight Details")
    flight_data = aviation_api_call(departure_iata, arrival_iata, flight_number if flight_number else None)
 
    if not flight_data or not flight_data.get('data'):
        st.info("No flights available at the moment.")  # Messaggio informativo per quando non ci sono voli
    else:
        # Creare una lista di dizionari per i dettagli dei voli
        flight_details = []
        for flight in flight_data['data']:
            # Convertire le date in formato locale e separare data e ora
            departure_time = convert_utc_to_local(flight['departure']['estimated'])
            arrival_time = convert_utc_to_local(flight['arrival']['estimated'])
           
            flight_details.append({
                "Flight Number": flight['flight']['iata'],
                "Airline": flight['airline']['name'],
                "Status": flight['flight_status'].capitalize(),  # Rendere leggibile lo status
                "Flight Date": departure_time.split()[0],  # Data del volo
                "Departure Time": departure_time.split()[1],  # Ora stimata di partenza
                "Arrival Time": arrival_time.split()[1],  # Ora stimata di arrivo
            })
       
        # Convertire la lista di dizionari in un DataFrame
        df_flights = pd.DataFrame(flight_details)
       
        # Visualizzare la tabella con larghezza piena
        st.dataframe(df_flights, use_container_width=True)
 
 