import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Otsikko
st.title("Fysiikan loppuprojekti")

# Määritellään tiedostojen polut
path_acceleration = "Linear Acceleration.csv"
path_location = "Location.csv"

# Luetaan kiihtyvyysdata ja GPS-data
accel_data = pd.read_csv(path_acceleration)
gps_data = pd.read_csv(path_location)

# Suodatetaan x-komponentti
window_size = 5  # Liukuva keskiarvo 5 datapisteelle
x_component = accel_data['Linear Acceleration x (m/s^2)']
filtered_x = x_component.rolling(window=window_size, min_periods=1).mean()

# Lasketaan askeleet suodatetusta datasta
threshold = 0.3  # Kynnysarvo askelien laskemiseen
peaks, _ = find_peaks(filtered_x, height=threshold)
steps_count_filtered = len(peaks)

# Lasketaan askeleet Fourier-analyysin avulla
fft_result = np.fft.fft(filtered_x.dropna())
magnitude = np.abs(fft_result)
frequencies = np.fft.fftfreq(len(magnitude))

# Lasketaan askelmäärä Fourier-analyysin perusteella
threshold_fft = 0.1 * np.max(magnitude)  # Kynnysarvo askelmäärälle
steps_count_fft = np.sum(magnitude > threshold_fft)

st.write("Askelmäärä laskettuna suodatuksen avulla: {:.1f} askelta".format(steps_count_filtered))
st.write("Askelmäärä laskettuna Fourier-analyysin avulla: {:.1f} askelta".format(steps_count_fft))

# Lasketaan GPS-data
lat_mean = gps_data['Latitude (°)'].mean()
long_mean = gps_data['Longitude (°)'].mean()
route_map = folium.Map(location=[lat_mean, long_mean], zoom_start=15)

coordinates = gps_data[['Latitude (°)', 'Longitude (°)']].values
folium.PolyLine(coordinates, color='blue', weight=2.5, opacity=1).add_to(route_map)

# Lasketaan matka Haversine-kaavalla
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c 

total_distance = 0
for i in range(1, len(gps_data)):
    lat1, lon1 = gps_data['Latitude (°)'].iloc[i - 1], gps_data['Longitude (°)'].iloc[i - 1]
    lat2, lon2 = gps_data['Latitude (°)'].iloc[i], gps_data['Longitude (°)'].iloc[i]
    total_distance += haversine(lat1, lon1, lat2, lon2)

total_distance_m = total_distance * 1000  # Muutetaan kilometrit metreiksi
total_time = gps_data['Time (s)'].iloc[-1] - gps_data['Time (s)'].iloc[0]
average_speed = total_distance_m / total_time

# Lasketaan askelpituus suodatetusta datasta
stride_length = total_distance_m / steps_count_filtered if steps_count_filtered > 0 else 0

# Lasketaan askelpituus Fourier-analyysin avulla lasketusta askelmäärästä
stride_length_fft = total_distance_m / steps_count_fft if steps_count_fft > 0 else 0

# Näytetään matka, keskinopeus ja askelpituus
st.write("Keskinopeus: {:.2f} m/s".format(average_speed))
st.write("Kokonaismatka: {:.2f} m".format(total_distance_m))
st.write("Askelpituus (suodatetusta datasta) on {:.2f} cm".format(stride_length * 100))
st.write("Askelpituus (Fourier-analyysin avulla) on {:.2f} cm".format(stride_length_fft * 100))

# Piirretään suodatettu signaali
st.subheader("Suodatetun kiihtyvyysdatan X-komponentti")
fig, ax = plt.subplots(figsize=(10, 4))
ax.grid()
ax.plot(accel_data['Time (s)'], filtered_x, color='orange')
ax.set_xlabel("Aika (s)")
ax.set_ylabel("Suodatettu X (m/s²)")
st.pyplot(fig)

# Tehospektri
st.subheader("Tehospektri")
plt.figure(figsize=(10, 4))
plt.grid()
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.xlabel("Taajuus (Hz)")
plt.ylabel("Teho")
st.pyplot(plt)

# Näytetään kartta Streamlitissä
st.subheader("Karttakuva")
st_data = st_folium(route_map, width=700, height=500)
