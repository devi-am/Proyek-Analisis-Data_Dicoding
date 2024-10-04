import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Change the sidebar width */
        .sidebar {
            width: 10px;
        }
        
        /* Adjust main content to fit the new sidebar size */
        .sidebar-content {
            padding-left: 10px;
        }
    </style>
    """, unsafe_allow_html=True)


# Header
st.title('Proyek Analisis Data: Bike Sharing Dataset')
st.write('By: Devita Ayu Maharani')
st.write('Email: mdevita303@gmail.com')

# Load dataset
@st.cache_data
def load_data():
    df_hour = pd.read_csv('data/hour.csv')
    df_day = pd.read_csv('data/day.csv')
    return df_hour, df_day

df_hour, df_day = load_data()

########################################################

# Cleaning Dataset
# tabel df_hour
df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])
print(df_hour.dtypes)

# tabel df_day
df_day['dteday'] = pd.to_datetime(df_day['dteday'])
print(df_day.dtypes)

#### Encoding Weathersit
# tabel df_hour
df_hour['weathersit'] = df_hour['weathersit'].replace({
    1: 'Clear/Few Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Snow/Rain',
    4: 'Heavy Snow/Rain'
})

print("Tipe data weathersit: ", df_hour['weathersit'].dtypes)
df_hour.head()

# tabel df_day
df_day['weathersit'] = df_day['weathersit'].replace({
    1: 'Clear/Few Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Snow/Rain',
    4: 'Heavy Snow/Rain'
})

print("Tipe data weathersit: ", df_day['weathersit'].dtypes)
df_day.head()

#### Encoding Season
# tabel df_hour
df_hour['season'] = df_hour['season'].replace({
    1: 'Springer',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter'
})

print("Tipe data season: ", df_hour['season'].dtypes)
df_hour.head()

# tabel df_day
df_day['season'] = df_day['season'].replace({
    1: 'Springer',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter'
})

print("Tipe data season: ", df_day['season'].dtypes)
df_day.head()

#### Drop instant
# tabel df_hour
df_hour = df_hour.drop(columns=['instant'])
df_hour.head()

# tabel df_day
df_day = df_day.drop(columns=['instant'])
df_day.head()

#### Rename header tabel
# tabel df_hour
df_hour = df_hour.rename(columns={
    'dteday': 'date',
    'yr': 'year',
    'mnth': 'month',
    'hr': 'hour',
    'weathersit': 'weather_condition',
    'temp': 'temperature',
    'atemp': 'temperature_feeling',
    'hum': 'humidity',
    'cnt': 'total_rented'
})
df_hour.head()

# tabel df_day
df_day = df_day.rename(columns={
    'dteday': 'date',
    'yr': 'year',
    'mnth': 'month',
    'weathersit': 'weather_condition',
    'temp': 'temperature',
    'atemp': 'temperature_feeling',
    'hum': 'humidity',
    'cnt': 'total_rented'
})
df_day.head()

########################################################

## Exploratory Data Analysis (EDA)

### Pola penggunaan sepeda selama 1 tahun
# tabel df_hour
# musim
print("Bedasarkan Musim:")
df_hour_season_rented = df_hour.groupby(by='season').agg({'total_rented': ['sum','mean','min','max']}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()
print(df_hour_season_rented)

# bulan
print("\nBedasarkan Bulan:")
df_hour_month_rented = df_hour.groupby(by='month').agg({'total_rented': ['sum','mean','min','max']}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()
print(df_hour_month_rented)

# jam
print("\nBedasarkan Jam:")
df_hour_hours_rented = df_hour.groupby(by='hour').agg({'total_rented': ['sum','mean','min','max']}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()
print(df_hour_hours_rented)

# jam per musim
df_hour_hours_season_rented = df_hour.groupby(by=['season', 'hour']).agg({'total_rented': ['sum','mean','min','max']}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()

# tabel df_day
# musim
print("Bedasarkan Musim:")
df_day_season_rented = df_day.groupby(by='season').agg({'total_rented': ['sum','mean','min','max']}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()
print(df_day_season_rented)

# bulan
print("\nBedasarkan Bulan:")
df_day_month_rented = df_day.groupby(by=['year','month']).agg({'total_rented': ['sum','mean','min','max']}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()

### Cuaca - Total rented
# tabel df_hour
df_hour_cor_weather_rented = df_hour[['temperature', 'temperature_feeling', 'humidity', 'windspeed', 'casual', 'registered', 'total_rented']].corr().round(2).reset_index()

# tabel df_day
df_day_cor_weather_rented = df_day[['temperature', 'temperature_feeling', 'humidity', 'windspeed', 'casual', 'registered', 'total_rented']].corr().round(2).reset_index()

# tabel df_hour
df_hour_weather_rented = df_hour.groupby(by='weather_condition').agg({
    'total_rented': ['sum','mean','min','max']
}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()

# tabel df_day
df_day_weather_rented = df_day.groupby(by='weather_condition').agg({
    'total_rented': ['sum','mean','min','max']
}).sort_values(by=('total_rented','sum'), ascending=False).round(2).reset_index()

### Hari libur/kerja - Total rented
# tabel df_hour
df_hour_workday_rented = df_hour.groupby(by='workingday').agg({
     'total_rented': ['sum','min','max']
}).sort_values(by=('total_rented','sum'), ascending=False).round(2)

df_hour_workday_rented = df_hour_workday_rented.reset_index()

day_type = {0:'holiday', 1:'workday'}
df_hour_workday_rented['workingday'] = df_hour_workday_rented['workingday'].map(day_type)

# tabel df_day
df_day_workday_rented = df_day.groupby(by='workingday').agg({
     'total_rented': ['sum','min','max'],
}).sort_values(by=('total_rented','sum'), ascending=False).round(2)

df_day_workday_rented = df_day_workday_rented.reset_index()

day_type = {0:'holiday', 1:'workday'}
df_day_workday_rented['workingday'] = df_day_workday_rented['workingday'].map(day_type)

df_day_workday_rented['mean_per_day'] = [df_day_workday_rented['total_rented']['sum'][0]/len(df_day[df_day['workingday'] == 1]),
                                          df_day_workday_rented['total_rented']['sum'][1]/len(df_day[df_day['workingday'] == 0])]

########################################################

# Sidebar Navigation
st.sidebar.title("Bike Sharing Data Analysis")
options = st.sidebar.selectbox("Choose a section", [
    "Data Wrangling", "Pola Penggunaan Sepeda", "Cuaca terhadap Sewa Sepeda", "Holiday vs Workday"
])

# Data Wrangling Section
if options == "Data Wrangling":
    st.subheader("Data Wrangling: Cleaning and Preparing Data")
    st.markdown("Sebelum melakukan analisis, dilakukan proses assessing serta cleaning data terlebih dahulu. Berikut adalah cuplikan data dari dua tabel data pada ***Bike Sharing Data***.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Tabel hour.csv")
        st.write(df_hour.head())
    
    with col2:
        st.markdown("#### Tabel day.csv")
        st.write(df_day.head())
    
    st.markdown("Kedua tabel telah di-_cleaning_ dengan: mengubah tipe data dteday menjadi ```datetime```, mengubah ***weathersit dan season*** ke bentuk ```string```, melakukan ***drop*** pada kolom ***insight***, dan me-_rename_ kolom tabel agar lebih mudah dibaca.")

# Pola Penggunaan Sepeda Section
if options == "Pola Penggunaan Sepeda":
    st.subheader("Pola Penggunaan Sepeda selama 2 Tahun")
    st.markdown("Jumlah sewa sepeda memiliki pola tiap tahunnya. Pola ini dapat dilihat dari tren sewa sepeda tiap musim, tren tiap sepeda tiap bulan, serta tren sewa sepeda perjamnya yang akan dibandingkan tiap musim. Berikut adalah grafik pemaparan analisis tersebut.")
    
    st.markdown("#### Perbandingan Jumlah Sewa Sepeda tiap Musimnya")
    
    col1, col2 = st.columns(2)
    with col1:
        # Grafik penyewaan sepeda per hari selama 2 tahun berdasarkan musim
        st.markdown("Penyewaan Sepeda per Hari")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

        sns.barplot(data=df_day_season_rented, x='season', y=('total_rented', 'sum'), palette=colors, ax=ax)
        ax.set_title('Penyewaan Sepeda per Hari Selama 2 Tahun Berdasarkan Musim')
        ax.set_xlabel('Musim')
        ax.set_ylabel('Jumlah Penyewaan')

        st.pyplot(fig)

    with col2:
        # Grafik penyewaan sepeda per jam selama 2 tahun berdasarkan musim
        st.markdown("Penyewaan Sepeda per Jam")
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(data=df_hour_season_rented, x='season', y=('total_rented', 'sum'), palette=colors, ax=ax)
        ax.set_title('Penyewaan Sepeda per Jam Selama 2 Tahun Berdasarkan Musim')
        ax.set_xlabel('Musim')
        ax.set_ylabel('Jumlah Penyewaan')

        st.pyplot(fig)
    
    # Grafik penyewaan sepeda selama 2 tahun berdasarkan bulan (total rented)
    df_sort_month_rented = df_day_month_rented.sort_values(by=['year', 'month'], ascending=True).reset_index(drop=True)
    df_sort_month_rented['year'] = df_sort_month_rented['year'].map({0: '2011', 1: '2012'})
    df_sort_month_rented['year_month'] = df_sort_month_rented['year'].astype(str) + '-' + df_sort_month_rented['month'].astype(str).str.zfill(2)

    st.markdown("### Penyewaan Sepeda Selama 2 Tahun tiap Bulannya")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=df_sort_month_rented, x='year_month', y=('total_rented', 'sum'), ax=ax)
    ax.set_title('Penyewaan Sepeda Selama 2 Tahun Berdasarkan Bulan')
    ax.set_xlabel('')
    ax.set_ylabel('Jumlah Penyewaan')
    ax.set_xticklabels(df_sort_month_rented['year_month'], rotation=45)
    st.pyplot(fig)

    # Grafik rata-rata, maksimum, minimum penyewaan sepeda per bulan
    st.markdown("### Rata-Rata Penyewaan Sepeda per Bulan Selama 2 Tahun")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=df_sort_month_rented, x='year_month', y=('total_rented', 'mean'), label='rata-rata', ax=ax)
    sns.lineplot(data=df_sort_month_rented, x='year_month', y=('total_rented', 'max'), label='maksimum', ax=ax)
    sns.lineplot(data=df_sort_month_rented, x='year_month', y=('total_rented', 'min'), label='minimum', ax=ax)
    ax.set_title('Penyewaan Sepeda Rata-Rata per Bulan Selama 2 Tahun')
    ax.set_xlabel('')
    ax.set_ylabel('Jumlah Penyewaan')
    ax.set_xticklabels(df_sort_month_rented['year_month'], rotation=45)
    ax.legend(title='keterangan')
    st.pyplot(fig)

    # Grafik pola penyewaan sepeda per jam berdasarkan musim
    df_sort_hours_season_rented = df_hour_hours_season_rented.sort_values(by='hour', ascending=True).reset_index(drop=True)
    df_sort_hours_season_rented['hour'] = df_sort_hours_season_rented['hour'].map({
        0: '00:00', 1: '01:00', 2: '02:00', 3: '03:00', 4: '04:00',
        5: '05:00', 6: '06:00', 7: '07:00', 8: '08:00', 9: '09:00',
        10: '10:00', 11: '11:00', 12: '12:00', 13: '13:00', 14: '14:00',
        15: '15:00', 16: '16:00', 17: '17:00', 18: '18:00', 19: '19:00',
        20: '20:00', 21: '21:00', 22: '22:00', 23: '23:00'
    })

    st.markdown("### Pola Penyewaan Sepeda per Jam Berdasarkan Musim")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = [f"{hour:02d}:00" for hour in range(24)]

    spring_data = df_sort_hours_season_rented[df_sort_hours_season_rented['season'] == 'Springer']
    sns.lineplot(data=spring_data, x='hour', y=('total_rented', 'sum'), label='Spring', ax=ax)

    summer_data = df_sort_hours_season_rented[df_sort_hours_season_rented['season'] == 'Summer']
    sns.lineplot(data=summer_data, x='hour', y=('total_rented', 'sum'), label='Summer', ax=ax)

    fall_data = df_sort_hours_season_rented[df_sort_hours_season_rented['season'] == 'Fall']
    sns.lineplot(data=fall_data, x='hour', y=('total_rented', 'sum'), label='Fall', ax=ax)

    winter_data = df_sort_hours_season_rented[df_sort_hours_season_rented['season'] == 'Winter']
    sns.lineplot(data=winter_data, x='hour', y=('total_rented', 'sum'), label='Winter', ax=ax)

    ax.set_title('Pola Penyewaan Sepeda per Jam Berdasarkan Musim')
    ax.set_xlabel('Waktu (Jam)')
    ax.set_ylabel('Jumlah Penyewaan')
    ax.set_xticklabels(x, rotation=45)
    ax.legend(title='Keterangan')
    st.pyplot(fig)
    
    st.markdown("Bedasarkan grafik grafik tersebut dapat disimpulkan bahwa tren sewa sepeda paling tinggi terjadi pada ***Musim Gugur***. Hal ini juga terlihat bahwa titik penyewaan tertinggi terjadi pada bulan 6 pada 2011 dan bulan 9 pada 2012 yang mana keduanya merupakan musim gugur.")
    st.markdown("Sedangkan jam penyewaan sepeda tertinggi memiliki kesamaan tiap musimnya. Dua titik sewa tertinggi terjadi pada jam 8 pagi dan jam 17 atau jam 5 sore.")

# Cuaca Sewa Sepeda Section
if options == "Cuaca terhadap Sewa Sepeda":
    st.subheader("Pengaruh Cuaca terhadap Jumlah Sewa Sepeda")
    st.markdown("Kondisi cuaca dapat memengaruhi jumlah sewa sepeda pada suatu waktu. Tingkat penyewaan sepeda dapat terlihat pada grafik korelasi serta perbandingan jumlah sewa tiap jenis cuaca berikut.")
    st.markdown("### Korelasi Temperatur dan Kelembapan dengan Penyewaan Sepeda")
    
    col1, col2 = st.columns(2)
    with col1:
        # Grafik korelasi antara suhu kelembapan dengan jumlah sewa per jam
        st.markdown("***Sewa Sepeda per Jam***")
        labels = df_hour_cor_weather_rented['index']
        df_hour_cor_weather_rented_numeric = df_hour_cor_weather_rented.drop(columns='index')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_hour_cor_weather_rented_numeric, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title('Korelasi Temperatur dan Kelembapan dengan Penyewaan Sepeda per Jam')
        st.pyplot(fig)

    with col2:
        # Grafik korelasi antara suhu kelembapan dengan jumlah sewa per hari
        st.markdown("***Sewa Sepeda per Hari***")
        labels = df_day_cor_weather_rented['index']
        df_day_cor_weather_rented_numeric = df_day_cor_weather_rented.drop(columns='index')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_day_cor_weather_rented_numeric, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title('Korelasi Temperatur dan Kelembapan dengan Penyewaan Sepeda per Hari')
        st.pyplot(fig)

    st.markdown("### Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca")
    
    col1, col2 = st.columns(2)
    with col1:
        # Grafik Jumlah Sewa per Jam berdasarkan Cuaca
        st.markdown("***Sewa Sepeda per Jam***")
        colors = ['#9BD5E8', '#8994A9', '#236EB8', '#10263C', '#FFFF00']

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_hour_weather_rented, x='weather_condition', y=('total_rented', 'sum'), palette=colors, ax=ax)
        ax.set_title('Jumlah Penyewaan Sepeda per Jam Berdasarkan Kondisi Cuaca')
        ax.set_xlabel('Kondisi Cuaca')
        ax.set_ylabel('Jumlah Penyewaan')
        ax.set_yscale('symlog')
        st.pyplot(fig)

    with col2:
        # Grafik Jumlah Sewa per Hari berdasarkan Cuaca
        st.markdown("***Sewa Sepeda per Hari***")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_day_weather_rented, x='weather_condition', y=('total_rented', 'sum'), palette=colors, ax=ax)
        ax.set_title('Jumlah Penyewaan Sepeda per Hari Berdasarkan Kondisi Cuaca')
        ax.set_xlabel('Kondisi Cuaca')
        ax.set_ylabel('Jumlah Penyewaan')
        ax.set_yscale('symlog')
        st.pyplot(fig)
    
    st.markdown("Bedasarkan grafik grafik tersebut dapat disimpulkan bahwa jumlah sewa sepeda memiliki korelasi positif (searah) dengan temperatur, sedangkan jumlah sewa sepeda memiliki kecenderungan korelasi negatif dengan humidity dan windspeed.")
    st.markdown("Sedangkan sepeda paling banyak dipinjam pada saat cuaca cerah atau sedikit berawan. Semakin buruk cuacanya, maka semakin sedikit jumlah peminjaman.")
    
# Holiday vs Workday Section
if options == "Holiday vs Workday":
    st.subheader("Jumlah Sewa Sepeda saat Holiday vs Workday")
    st.markdown("Apabila kita membandingkan jumlah sewa sepeda saat hari libur dan hari kerja maka kita akan menemukan perbedaan. Berikut adalah perbandingan antara sewa sepeda hari libur dan hari kerja")
    
    # Grafik perbandingan sewa sepeda saat Hari Libur dan Kerja
    fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
    colors = ["#f2a33a", "#eadec3"]

    # Barplot 1: Total rental comparison between working and non-working days
    sns.barplot(data=df_day_workday_rented, x='workingday', y=('total_rented','sum'), palette=colors, ax=ax[0])
    ax[0].set_title('Penyewaan Sepeda Berdasarkan Hari Kerja/Libur')
    ax[0].set_xlabel('Hari Kerja/Libur')
    ax[0].set_ylabel('Jumlah Penyewaan')

    # Barplot 2: Average rental comparison between working and non-working days
    sns.barplot(data=df_day_workday_rented, x='workingday', y='mean_per_day', palette=colors, ax=ax[1])
    ax[1].set_title('Rata-Rata Penyewaan Sepeda Berdasarkan Hari Kerja/Libur')
    ax[1].set_xlabel('Hari Kerja/Libur')
    ax[1].set_ylabel('Jumlah Penyewaan')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.markdown("Bedasarkan tabel tersebut dapat disimpulkan bahwa Hari ***Kerja memiliki*** tingkat sewa yang lebih tinggi dari Hari Libur. Hal ini mungkin berhubungan dengan waktu (jam) tertinggi penyewaan pada jam 8 pagi dan jam 5 sore yang mana hal tersebut merupakan rata rata jam berangkat dan pulang kerja.")