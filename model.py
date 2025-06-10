import streamlit as st
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.ma.core import left_shift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import json
from sklearn.metrics import mean_absolute_error

# --------------------------------
# Stil
# --------------------------------
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #fff7e6, #ffd7b5);
        border-right: 2px solid #ffcc80;
        box-shadow: 2px 0px 5px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------
# ✅ Cache'li Model Eğitim Fonksiyonu
# --------------------------------
@st.cache_resource
def train_models_once():
    # -------------------------
    # Kiralık Model Eğitimi
    # -------------------------
    df = pd.read_csv("hepsiemlakpreprocessing.csv", encoding="utf-8")
    le_konut = LabelEncoder()
    df["Konut Tipi Encoded"] = le_konut.fit_transform(df["Konut Tipi"])
    konut_mapping = dict(zip(le_konut.classes_, le_konut.transform(le_konut.classes_)))
    y = df["Fiyat"]
    X = df[[
        "Konut Tipi Encoded", "Oda Sayısı", "Kat", "Net m2", "Brüt m2", "Kat Sayısı",
        "Bina Yaşı", "Banyo Sayısı", "Eşya Durumu", "Isınma Tipi Encoded",
        "Yapının Durumu Encoded", "Yapı Tipi Encoded", "Yakıt Tipi Encoded",
        "İlçe Encoded", "Mahalle Encoded", "Aidat", "Depozito",
        "Cephe_Kuzey", "Cephe_Güney", "Cephe_Doğu", "Cephe_Batı", "Cephe_Bilinmiyor"
    ]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=144)
    model = LGBMRegressor(learning_rate=0.1, max_depth=7, n_estimators=500,
                          colsample_bytree=0.6, min_child_samples=10, min_child_weight=0.001, num_leaves=30)
    model.fit(X_train, y_train)
    model_mae = mean_absolute_error(y_test, model.predict(X_test))

    # -------------------------
    # Satılık Model Eğitimi
    # -------------------------
    df2 = pd.read_csv("satılıkpreprocessing.csv", encoding="utf-8")
    le_konut2 = LabelEncoder()
    df2["Konut Tipi Encoded"] = le_konut2.fit_transform(df2["Konut Tipi"])
    konut_mapping2 = dict(zip(le_konut2.classes_, le_konut2.transform(le_konut2.classes_)))
    y2 = df2["Fiyat"]
    X2 = df2[[
        "Konut Tipi Encoded", "Oda Sayısı", "Kat", "Net m2", "Brüt m2", "Kat Sayısı",
        "Bina Yaşı", "Banyo Sayısı", "Eşya Durumu", "Isınma Tipi Encoded",
        "Yapının Durumu Encoded", "Yapı Tipi Encoded", "Yakıt Tipi Encoded",
        "İlçe Encoded", "Mahalle Encoded", "Krediye Uygunluk",
        "Cephe_Kuzey", "Cephe_Güney", "Cephe_Doğu", "Cephe_Batı", "Cephe_Bilinmiyor"
    ]]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
    model2 = LGBMRegressor(
        learning_rate=0.1, max_depth=-1, n_estimators=500, colsample_bytree=0.8,
        min_child_samples=5, num_leaves=70, reg_alpha=0.1, reg_lambda=1, subsample=0.6
    )
    model2.fit(X_train2, y_train2)
    model2_mae = mean_absolute_error(y_test2, model2.predict(X_test2))

    return model, model_mae, model2, model2_mae, df, df2 , konut_mapping, konut_mapping2 ,X, X2, X_test, y_test ,X_test2 ,y_test2

# --------------------------------
# Model ve verileri sadece bir kez eğit ve yükle
# --------------------------------
model, model_mae, model2, model2_mae, df, df2 , konut_mapping, konut_mapping2 ,X, X2, X_test, y_test ,X_test2 ,y_test2 = train_models_once()

# --------------------------------
# Streamlit Arayüzü
# --------------------------------
if "page" not in st.session_state:
    st.session_state.page = "ℹ️ Home Page"

selected_page = st.sidebar.radio("Page Selection", [
    "🏠 Rent Prediction",
    "🏡 Sale Prediction",
    "📈 Data Visualization",
    "ℹ️ Home Page",
    "🗺️ Heatmap"
], index=[
    "🏠 Rent Prediction",
    "🏡 Sale Prediction",
    "📈 Data Visualization",
    "ℹ️ Home Page",
    "🗺️ Heatmap"
].index(st.session_state.page))

# Update session state
st.session_state.page = selected_page

if st.session_state.page != "ℹ️ Home Page":
    st.title("🏠 BEstate - Istanbul Housing Price Prediction System")

# Page 1 - Rent Price Prediction
# --------------------------------

if st.session_state.page == "🏠 Rent Prediction":
    st.header("📊 Rent Price Estimation")

    try:
        logo = Image.open("BEstate.png")
        st.image(logo, width=300)
    except:
        st.warning("Logo not found. Make sure you have uploaded 'BEstate.png'.")

    # 🔁 Mapping dictionaries
    heating_type_mapping = {
        "Diğer": "Other",
        "Fancoil Ünitesi": "Fan Coil Unit",
        "Isıtma Yok": "No Heating",
        "Kat Kaloriferi": "Floor Radiator",
        "Klima": "Air Conditioner",
        "Kombi": "Combi Boiler",
        "Merkezi": "Central",
        "Soba": "Stove",
        "Yerden Isıtma": "Underfloor Heating"
    }

    fuel_type_mapping = {
        "Diğer": "Other",
        "Doğalgaz": "Natural Gas",
        "Elektrik": "Electric"
    }

    building_type_mapping = {
        "Betonarme": "Reinforced Concrete",
        "Diğer": "Other"
    }

    building_condition_mapping = {
        "Diğer": "Other",
        "Sıfır": "New",
        "İkinci El": "Second-hand"
    }

    property_type_mapping = {
        "Daire": "Apartment",
        "Diğer": "Other",
        "Residence": "Residence",
        "Villa": "Villa"
    }

    furnishing_mapping = {
        "Eşyasız": "Unfurnished",
        "Eşyalı": "Furnished"
    }

    # Sidebar Inputs
    st.sidebar.markdown("### 🏠 Enter Rental Property Details")
    district = st.sidebar.selectbox("District", sorted(df["İlçe"].unique()))
    neighborhood = st.sidebar.selectbox("Neighborhood", sorted(df[df["İlçe"] == district]["Mahalle"].unique()))

    property_label = st.sidebar.selectbox(
        "Property Type",
        options=[property_type_mapping[val] for val in df["Konut Tipi"].unique()]
    )
    property_type = [k for k, v in property_type_mapping.items() if v == property_label][0]
    apartment_type = konut_mapping[property_type]

    num_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, value=2)
    floor = st.sidebar.number_input("Floor", min_value=-3, max_value=50, value=1)
    net_m2 = st.sidebar.number_input("Net Area (m²)", min_value=30, max_value=400, value=100)
    brut_m2 = st.sidebar.number_input("Gross Area (m²)", min_value=30, max_value=500, value=120)
    total_floors = st.sidebar.number_input("Total Floors", min_value=1, max_value=50, value=5)
    building_age = st.sidebar.number_input("Building Age", min_value=0, max_value=150, value=10)
    num_bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=0, max_value=5, value=1)

    furnishing_label = st.sidebar.selectbox("Furnishing", options=list(furnishing_mapping.values()))
    furnishing_value = [k for k, v in furnishing_mapping.items() if v == furnishing_label][0]
    furnished_encoded = 1 if furnishing_value == "Eşyalı" else 0

    heating_label = st.sidebar.selectbox(
        "Heating Type",
        options=[heating_type_mapping[val] for val in sorted(df["Isınma Tipi"].unique())]
    )
    heating = [k for k, v in heating_type_mapping.items() if v == heating_label][0]
    heating_encoded = int(df[df["Isınma Tipi"] == heating]["Isınma Tipi Encoded"].iloc[0])

    structure_status_label = st.sidebar.selectbox(
        "Building Condition",
        options=[building_condition_mapping[val] for val in sorted(df["Yapının Durumu"].unique())]
    )
    structure_status = [k for k, v in building_condition_mapping.items() if v == structure_status_label][0]
    structure_status_encoded = int(df[df["Yapının Durumu"] == structure_status]["Yapının Durumu Encoded"].iloc[0])

    structure_type_label = st.sidebar.selectbox(
        "Building Type",
        options=[building_type_mapping[val] for val in sorted(df["Yapı Tipi"].unique())]
    )
    structure_type = [k for k, v in building_type_mapping.items() if v == structure_type_label][0]
    structure_type_encoded = int(df[df["Yapı Tipi"] == structure_type]["Yapı Tipi Encoded"].iloc[0])

    fuel_type_label = st.sidebar.selectbox(
        "Fuel Type",
        options=[fuel_type_mapping[val] for val in df["Yakıt Tipi"].unique()]
    )
    fuel_type = [k for k, v in fuel_type_mapping.items() if v == fuel_type_label][0]
    fuel_type_encoded = int(df[df["Yakıt Tipi"] == fuel_type]["Yakıt Tipi Encoded"].iloc[0])

    aidat = st.sidebar.number_input("Dues (TL)", min_value=0, max_value=50000, value=500)
    deposit = st.sidebar.number_input("Deposit (TL)", min_value=0, max_value=500000, value=10000)

    facade = st.sidebar.multiselect("Orientation", ["North", "South", "East", "West"])
    cephe_kuzey = 1 if "North" in facade else 0
    cephe_guney = 1 if "South" in facade else 0
    cephe_dogu = 1 if "East" in facade else 0
    cephe_bati = 1 if "West" in facade else 0
    cephe_bilinmiyor = 1 if len(facade) == 0 else 0

    district_encoded = df[df["İlçe"] == district]["İlçe Encoded"].values[0]
    neighborhood_encoded = df[df["Mahalle"] == neighborhood]["Mahalle Encoded"].values[0]

    input_data = np.array([[
        apartment_type, num_rooms, floor, net_m2, brut_m2, total_floors, building_age, num_bathrooms,
        furnished_encoded, heating_encoded, structure_status_encoded, structure_type_encoded, fuel_type_encoded,
        district_encoded, neighborhood_encoded, aidat, deposit,
        cephe_kuzey, cephe_guney, cephe_dogu, cephe_bati, cephe_bilinmiyor
    ]])


    if st.sidebar.button("🎯 Estimate Price"):
        prediction = model.predict(input_data)[0]
        formatted_price = f"{prediction:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # 1️⃣ Success Message
        st.success("✅ Prediction successfully calculated!")

        # 2️⃣ Prediction Range
        lower_bound = prediction - model_mae
        upper_bound = prediction + model_mae

        formatted_lower = f"{lower_bound:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        formatted_upper = f"{upper_bound:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # 3️⃣ Premium Styled Prediction Box
        st.markdown("""
            <style>
            .premium-price-box {
                background: linear-gradient(135deg, #f0fff4, #c6f6d5);
                border-left: 6px solid #38a169;
                padding: 25px;
                margin-top: 20px;
                margin-bottom: 20px;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                gap: 20px;
            }
            .premium-price-icon {
                font-size: 50px;
            }
            .premium-price-info {
                display: flex;
                flex-direction: column;
            }
            .premium-price-info h2 {
                color: #22543D;
                font-size: 40px;
                font-weight: bold;
                margin: 0;
            }
            .premium-price-info p {
                margin: 6px 0 0 0;
                color: #2D3748;
                font-size: 18px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="premium-price-box">
                <div class="premium-price-icon">💵</div>
                <div class="premium-price-info">
                    <h2>{formatted_price} TL</h2>
                    <p>This is the estimated monthly rent for this property.</p>
                    <p>~ May range between {formatted_lower} TL and {formatted_upper} TL.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 4️⃣ Summary of Inputs
        st.markdown("### 📝 Your Input Summary")
        st.markdown(f"""
            - **District:** {district}
            - **Neighborhood:** {neighborhood}
            - **Property Type:** {property_label}
            - **Number of Rooms:** {num_rooms}
            - **Floor:** {floor}
            - **Net Area (m²):** {net_m2}
            - **Gross Area (m²):** {brut_m2}
            - **Building Age:** {building_age}
            - **Number of Bathrooms:** {num_bathrooms}
            - **Furnishing:** {"Furnished" if furnished_encoded == 1 else "Unfurnished"}
            - **Heating Type:** {heating_label}
            - **Building Condition:** {structure_status_label}
            - **Building Type:** {structure_type_label}
            - **Fuel Type:** {fuel_type_label}
            - **Dues:** {aidat} TL
            - **Deposit:** {deposit} TL
            - **Orientation:** {', '.join(facade) if facade else "Unknown"}
        """)
        # 5️⃣ SHAP Values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=X.columns))

        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "SHAP Value": shap_values[0]
        }).sort_values(by="SHAP Value", key=np.abs, ascending=False).head(5)

        st.markdown("""
            <style>
            .shap-card {
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
            }
            .shap-card-title {
                font-weight: 700;
                font-size: 17px;
                margin-bottom: 6px;
            }
            .shap-card-desc {
                font-size: 15px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("### 📌 Most Influential Factors")

        # Icon matching for each feature
        feature_icons = {
            "Net m2": "📏",
            "Kat": "🏢",
            "Konut Tipi Encoded": "🏠",
            "İlçe Encoded": "🗺️",
            "Mahalle Encoded": "📍",
            "Aidat": "💳",
            "Depozito": "💰",
            "Bina Yaşı": "🏗️",
            "Isınma Tipi Encoded": "🔥",
            "Brüt m2": "📐",
            "Yapının Durumu Encoded": "🏚️",
            "Yakıt Tipi Encoded": "⛽",
            "Yapı Tipi Encoded": "🏡",
            "Cephe_Kuzey": "⬆️",
            "Cephe_Güney": "⬇️",
            "Cephe_Batı": "⬅️",
            "Cephe_Doğu": "➡️"
        }

        for _, row in shap_df.iterrows():
            effect = "increased" if row["SHAP Value"] > 0 else "decreased"
            bg_color = "#C6F6D5" if row["SHAP Value"] > 0 else "#FED7D7"
            border_color = "#2F855A" if row["SHAP Value"] > 0 else "#C53030"
            text_color = "#22543D" if row["SHAP Value"] > 0 else "#742A2A"
            formatted_shap = f"{abs(row['SHAP Value']):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
            icon = feature_icons.get(row['Feature'], "✨")

            st.markdown(f"""
                <div class='shap-card' style='background-color:{bg_color}; border-left: 6px solid {border_color};'>
                    <div class='shap-card-title' style='color:{text_color}'>{icon} {row['Feature']}</div>
                    <div class='shap-card-desc' style='color:#4A5568;'>
                        This feature <strong>{effect}</strong> the estimated rent by approximately 
                        <strong>{formatted_shap} TL</strong>.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # 6️⃣ SHAP Waterfall Chart
        st.markdown("### 🔍 Feature Impact Chart")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=X.columns,
                                               max_display=5, show=False)
        st.pyplot(fig)

# --------------------------------
# Page 2 - Sale Price Prediction
# --------------------------------

elif st.session_state.page == "🏡 Sale Prediction":
    st.header("📊 House Sale Price Estimation")

    try:
        logo = Image.open("BEstate.png")
        st.image(logo, width=300)
    except:
        st.warning("Logo not found. Make sure you have uploaded 'BEstate.png'.")

    # 🔁 Mapping dictionaries
    heating_type_mapping = {
        "Diğer": "Other",
        "Fancoil Ünitesi": "Fan Coil Unit",
        "Isıtma Yok": "No Heating",
        "Kat Kaloriferi": "Floor Radiator",
        "Klima": "Air Conditioner",
        "Kombi": "Combi Boiler",
        "Merkezi": "Central",
        "Soba": "Stove",
        "Yerden Isıtma": "Underfloor Heating"
    }

    fuel_type_mapping = {
        "Diğer": "Other",
        "Doğalgaz": "Natural Gas",
        "Elektrik": "Electric"
    }

    building_type_mapping = {
        "Betonarme": "Reinforced Concrete",
        "Diğer": "Other"
    }

    building_condition_mapping = {
        "Diğer": "Other",
        "Sıfır": "New",
        "İkinci El": "Second-hand"
    }

    property_type_mapping = {
        "Daire": "Apartment",
        "Diğer": "Other",
        "Residence": "Residence",
        "Villa": "Villa"
    }

    furnishing_mapping = {
        "Eşyasız": "Unfurnished",
        "Eşyalı": "Furnished"
    }

    # Sidebar inputs
    st.sidebar.markdown("### 🏘️ Enter Sale Property Details")
    district = st.sidebar.selectbox("District", sorted(df2["İlçe"].unique()))
    neighborhood = st.sidebar.selectbox("Neighborhood", sorted(df2[df2["İlçe"] == district]["Mahalle"].unique()))

    property_label = st.sidebar.selectbox(
        "Property Type",
        options=[property_type_mapping[val] for val in df2["Konut Tipi"].unique()]
    )
    property_type = [k for k, v in property_type_mapping.items() if v == property_label][0]
    apartment_type = konut_mapping2[property_type]

    num_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, value=2)
    floor = st.sidebar.number_input("Floor", min_value=-3, max_value=50, value=1)
    net_m2 = st.sidebar.number_input("Net Area (m²)", min_value=30, max_value=400, value=100)
    brut_m2 = st.sidebar.number_input("Gross Area (m²)", min_value=30, max_value=500, value=120)
    total_floors = st.sidebar.number_input("Total Floors", min_value=1, max_value=50, value=5)
    building_age = st.sidebar.number_input("Building Age", min_value=0, max_value=150, value=10)
    num_bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=0, max_value=5, value=1)

    furnishing_label = st.sidebar.selectbox("Furnishing", options=list(furnishing_mapping.values()))
    furnishing_value = [k for k, v in furnishing_mapping.items() if v == furnishing_label][0]
    furnished_encoded = 1 if furnishing_value == "Eşyalı" else 0

    heating_label = st.sidebar.selectbox(
        "Heating Type",
        options=[heating_type_mapping[val] for val in sorted(df2["Isınma Tipi"].unique())]
    )
    heating = [k for k, v in heating_type_mapping.items() if v == heating_label][0]
    heating_encoded = int(df2[df2["Isınma Tipi"] == heating]["Isınma Tipi Encoded"].iloc[0])

    structure_status_label = st.sidebar.selectbox(
        "Building Condition",
        options=[building_condition_mapping[val] for val in sorted(df2["Yapının Durumu"].unique())]
    )
    structure_status = [k for k, v in building_condition_mapping.items() if v == structure_status_label][0]
    structure_status_encoded = int(df2[df2["Yapının Durumu"] == structure_status]["Yapının Durumu Encoded"].iloc[0])

    structure_type_label = st.sidebar.selectbox(
        "Building Type",
        options=[building_type_mapping[val] for val in sorted(df2["Yapı Tipi"].unique())]
    )
    structure_type = [k for k, v in building_type_mapping.items() if v == structure_type_label][0]
    structure_type_encoded = int(df2[df2["Yapı Tipi"] == structure_type]["Yapı Tipi Encoded"].iloc[0])

    fuel_type_label = st.sidebar.selectbox(
        "Fuel Type",
        options=[fuel_type_mapping[val] for val in df2["Yakıt Tipi"].unique()]
    )
    fuel_type = [k for k, v in fuel_type_mapping.items() if v == fuel_type_label][0]
    fuel_type_encoded = int(df2[df2["Yakıt Tipi"] == fuel_type]["Yakıt Tipi Encoded"].iloc[0])

    kredi = st.sidebar.selectbox("Mortgage Eligible?", ["Yes", "No"])
    kredi_encoded = 1 if kredi == "Yes" else 0

    facade = st.sidebar.multiselect("Orientation", ["North", "South", "East", "West"])
    cephe_kuzey = 1 if "North" in facade else 0
    cephe_guney = 1 if "South" in facade else 0
    cephe_dogu = 1 if "East" in facade else 0
    cephe_bati = 1 if "West" in facade else 0
    cephe_bilinmiyor = 1 if len(facade) == 0 else 0

    district_encoded = df2[df2["İlçe"] == district]["İlçe Encoded"].values[0]
    neighborhood_encoded = df2[df2["Mahalle"] == neighborhood]["Mahalle Encoded"].values[0]

    input_data = np.array([[
        apartment_type, num_rooms, floor, net_m2, brut_m2, total_floors, building_age, num_bathrooms,
        furnished_encoded, heating_encoded, structure_status_encoded, structure_type_encoded, fuel_type_encoded,
        district_encoded, neighborhood_encoded, kredi_encoded,
        cephe_kuzey, cephe_guney, cephe_dogu, cephe_bati, cephe_bilinmiyor
    ]])


    if st.sidebar.button("🎯 Predict Sale Price"):
        prediction = model2.predict(input_data)[0]

        # ✅ Success Message
        st.success("✅ Prediction successfully calculated!")

        # 🔄 Price Range
        lower_bound = prediction - model2_mae
        upper_bound = prediction + model2_mae
        formatted_lower = f"{lower_bound:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        formatted_upper = f"{upper_bound:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        formatted_price = f"{prediction:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # 💰 Premium Price Box (Including Range)
        st.markdown("""
            <style>
            .premium-price-box {
                background: linear-gradient(135deg, #e6fffa, #b2f5ea);
                border-left: 6px solid #319795;
                padding: 25px;
                margin-top: 20px;
                margin-bottom: 20px;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                gap: 20px;
            }
            .premium-price-icon {
                font-size: 50px;
            }
            .premium-price-info {
                display: flex;
                flex-direction: column;
            }
            .premium-price-info h2 {
                color: #1D4044;
                font-size: 40px;
                font-weight: bold;
                margin: 0;
            }
            .premium-price-info p {
                margin: 6px 0 0 0;
                color: #2D3748;
                font-size: 18px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="premium-price-box">
                <div class="premium-price-icon">💰</div>
                <div class="premium-price-info">
                    <h2>{formatted_price} TL</h2>
                    <p>This is the estimated sale price of the property.</p>
                    <p>~ It may range between {formatted_lower} TL and {formatted_upper} TL.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 📝 Input Summary
        st.markdown("### 📝 Your Input Summary")
        st.markdown(f"""
        - **District:** {district}  
        - **Neighborhood:** {neighborhood}  
        - **Property Type:** {property_label}  
        - **Number of Rooms:** {num_rooms}  
        - **Floor:** {floor}  
        - **Net Area (m²):** {net_m2}  
        - **Gross Area (m²):** {brut_m2}  
        - **Building Age:** {building_age}  
        - **Number of Bathrooms:** {num_bathrooms}  
        - **Furnishing:** {"Furnished" if furnished_encoded else "Unfurnished"}  
        - **Heating Type:** {heating_label}  
        - **Building Condition:** {structure_status_label}  
        - **Building Type:** {structure_type_label}  
        - **Fuel Type:** {fuel_type_label}  
        - **Mortgage Eligibility:** {kredi}  
        - **Orientation:** {', '.join(facade) if facade else 'Unknown'}  
        """)

        # 📌 SHAP Values
        explainer2 = shap.TreeExplainer(model2)
        shap_values2 = explainer2.shap_values(pd.DataFrame(input_data, columns=X2.columns))

        shap_df = pd.DataFrame({
            "Feature": X2.columns,
            "SHAP Value": shap_values2[0]
        }).sort_values(by="SHAP Value", key=np.abs, ascending=False).head(5)

        st.markdown("""
            <style>
            .shap-card {
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
            }
            .shap-card-title {
                font-weight: 700;
                font-size: 17px;
                margin-bottom: 6px;
            }
            .shap-card-desc {
                font-size: 15px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("### 📌 Most Influential Factors")

        feature_icons = {
            "Net m2": "📏",
            "Kat": "🏢",
            "Konut Tipi Encoded": "🏠",
            "İlçe Encoded": "🗺️",
            "Mahalle Encoded": "📍",
            "Bina Yaşı": "🏗️",
            "Isınma Tipi Encoded": "🔥",
            "Brüt m2": "📐",
            "Yapının Durumu Encoded": "🏚️",
            "Yakıt Tipi Encoded": "⛽",
            "Yapı Tipi Encoded": "🏡",
            "Cephe_Kuzey": "⬆️",
            "Cephe_Güney": "⬇️",
            "Cephe_Batı": "⬅️",
            "Cephe_Doğu": "➡️"
        }

        for _, row in shap_df.iterrows():
            effect = "increased" if row["SHAP Value"] > 0 else "decreased"
            bg_color = "#C6F6D5" if row["SHAP Value"] > 0 else "#FED7D7"
            border_color = "#2F855A" if row["SHAP Value"] > 0 else "#C53030"
            text_color = "#22543D" if row["SHAP Value"] > 0 else "#742A2A"
            formatted_shap = f"{abs(row['SHAP Value']):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
            icon = feature_icons.get(row['Feature'], "✨")

            st.markdown(f"""
                <div class='shap-card' style='background-color:{bg_color}; border-left: 6px solid {border_color};'>
                    <div class='shap-card-title' style='color:{text_color}'>{icon} {row['Feature']}</div>
                    <div class='shap-card-desc' style='color:#4A5568;'>
                        This feature <strong>{effect}</strong> the estimated sale price by approximately 
                        <strong>{formatted_shap} TL</strong>.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # 🔍 SHAP Chart
        st.markdown("### 🔍 Feature Impact Chart")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots._waterfall.waterfall_legacy(
            explainer2.expected_value,
            shap_values2[0],
            feature_names=X2.columns,
            max_display=5,
            show=False
        )
        st.pyplot(fig)
# Page 3 - Data Visualization
elif st.session_state.page == "📈 Data Visualization":
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("📊 Data Analysis and Visualizations")

    veri_turu = st.radio("Select Data Type:", ["Rent", "Sale"], horizontal=True)

    if veri_turu == "Rent":
        veri = df.copy()
        aktif_model = model
        X_test_v = X_test
        y_test_v = y_test
        baslik = "Rental Property Data"
        ek_sutunlar = ["Aidat", "Depozito"]
    else:
        veri = df2.copy()
        aktif_model = model2
        X_test_v = X_test2
        y_test_v = y_test2
        baslik = "Property for Sale Data"
        ek_sutunlar = ["Krediye Uygunluk"]

    # -----------------------------
    # Correlation Matrix
    # -----------------------------
    st.subheader(f"🔍 Correlation Matrix ({baslik})")

    corr_cols = [
        "Konut Tipi Encoded", "Oda Sayısı", "Kat", "Net m2", "Brüt m2",
        "Kat Sayısı", "Bina Yaşı", "Banyo Sayısı", "Eşya Durumu", "Isınma Tipi Encoded",
        "Yapının Durumu Encoded", "Yapı Tipi Encoded", "Yakıt Tipi Encoded",
        "İlçe Encoded", "Mahalle Encoded", "Cephe_Kuzey", "Cephe_Güney",
        "Cephe_Doğu", "Cephe_Batı", "Cephe_Bilinmiyor", "Fiyat"
    ] + ek_sutunlar

    corr_matrix = veri[corr_cols].corr()
    cols = [col for col in corr_matrix.columns if col != "Fiyat"] + ["Fiyat"]
    corr_matrix = corr_matrix.loc[cols, cols]

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    st.pyplot(fig)

    # -----------------------------
    # Actual vs Predicted Prices (Final - Only Image-Based)
    # -----------------------------
    st.subheader("📊 Model Prediction Error Analysis")

    plot_option = st.selectbox("Select Visualization Type:", [
        "Actual vs Predicted (Scatter)",
        "Absolute Error vs Predicted Price",
        "Residuals vs Predicted Price",
        "Distribution of Prediction Errors"
    ])

    file_map_rent = {
        "Actual vs Predicted (Scatter)": "kiralık_actualpredicted.png",
        "Absolute Error vs Predicted Price": "kiralık_absoluteerrorvspredictedprice.png",
        "Residuals vs Predicted Price": "kiralık_residualsvspredicted.png",
        "Distribution of Prediction Errors": "kiralık_distributionpredictionerror.png"
    }

    file_map_sale = {
        "Actual vs Predicted (Scatter)": "satılık_actualpredicted.png",
        "Absolute Error vs Predicted Price": "satılık_absoluterror.png",
        "Residuals vs Predicted Price": "satılık_residualsvspredicted.png",
        "Distribution of Prediction Errors": "satılık_residualerrordistribution.png"
    }

    selected_file = file_map_rent[plot_option] if veri_turu == "Rent" else file_map_sale[plot_option]

    try:
        st.image(selected_file, caption=f"{veri_turu} - {plot_option}", use_container_width=True)
    except:
        st.warning(f"{selected_file} not found. Please upload it.")

    # -----------------------------
    # Price Distribution
    # -----------------------------
    st.subheader("💰 Price Distribution")
    fiyat_secim = st.selectbox("Select Analysis Type:", [
        "Average Price by District",
        "Price by Property Type",
        "Price by Area (m²)",
        "Price by Number of Rooms",
        "Price by Building Age"
    ])

    if fiyat_secim == "Average Price by District":
        ort = veri.groupby("İlçe")["Fiyat"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        ort.plot(kind="barh", ax=ax)
        ax.set_title("Average Price by District")
        ax.set_xlabel("Price (TL)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
        st.pyplot(fig)

    elif fiyat_secim == "Price by Property Type":
        ort_fiyatlar = veri.groupby("Konut Tipi")["Fiyat"].mean().sort_values(ascending=False)
        st.bar_chart(ort_fiyatlar)

    elif fiyat_secim == "Price by Area (m²)":
        veri['Net m2 Bin'] = pd.cut(veri['Net m2'], bins=10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Net m2 Bin', y='Fiyat', data=veri, ax=ax)
        ax.set_title("Price Distribution by Net Area Range")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif fiyat_secim == "Price by Number of Rooms":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Oda Sayısı', y='Fiyat', data=veri, ax=ax)
        ax.set_title("Price Distribution by Number of Rooms")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
        st.pyplot(fig)

    elif fiyat_secim == "Price by Building Age":
        veri['Bina Yaşı Bin'] = pd.cut(veri['Bina Yaşı'], bins=10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Bina Yaşı Bin', y='Fiyat', data=veri, ax=ax)
        ax.set_title("Price Distribution by Building Age")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("📌 Feature Importance Ranking")

    importance_df = pd.DataFrame({
        "Feature": X_test_v.columns if hasattr(X_test_v, "columns") else X_test_v.columns_,
        "Importance": aktif_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance_df, y="Feature", x="Importance", palette="viridis", ax=ax)
    ax.set_title("LightGBM Feature Importances")
    st.pyplot(fig)

    # -----------------------------
    # Model Comparison Section
    # -----------------------------
    st.subheader("📈 Model Performance Comparison")

    comparison_metric = st.selectbox("Select Comparison Metric:", [
        "R² Score",
        "Explained Variance Score",
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Percentage Error (MAPE)",
        "Median Absolute Error",
        "Maximum Error"
    ])

    metric_file_map_rent = {
        "R² Score": "kiralık_r^2.png",
        "Explained Variance Score": "kiralık_explainedvariance.png",
        "Mean Absolute Error (MAE)": "kiralık_mae.png",
        "Root Mean Squared Error (RMSE)": "kiralık_rmse.png",
        "Mean Absolute Percentage Error (MAPE)": "kiralık_mape.png",
        "Median Absolute Error": "kiralık_medianae.png",
        "Maximum Error": "kiralık_maxerr.png"
    }

    metric_file_map_sale = {
        "R² Score": "satılık_r2.png",
        "Explained Variance Score": "satılık_explainedvariance.png",
        "Mean Absolute Error (MAE)": "satılık_maekarsılastırma.png",
        "Root Mean Squared Error (RMSE)": "satılık_rmsekarsılastırma.png",
        "Mean Absolute Percentage Error (MAPE)": "satılık_mapekarsılastırma.png",
        "Median Absolute Error": "satılık_medianae.png",
        "Maximum Error": "satılık_maxerror.png"
    }

    selected_file = metric_file_map_rent[comparison_metric] if veri_turu == "Rent" else metric_file_map_sale[
        comparison_metric]

    try:
        st.image(selected_file, caption=f"{veri_turu} Model Comparison - {comparison_metric}", use_container_width=True)
    except:
        st.warning(f"{selected_file} not found. Please upload it.")

    # -----------------------------
    # Model All Metrics Comparison
    # -----------------------------
    st.subheader("📊 Model All Metrics Comparison")

    if veri_turu == "Rent":
        metrics_path = "Modeller.png"
    else:
        metrics_path = "SatilikModeller.png"

    try:
        st.image(metrics_path, caption=f"{veri_turu} - Full Model Metrics Table", use_container_width=True)
    except:
        st.warning(f"{metrics_path} not found. Please upload it.")


# Page 4 - Home Page
elif st.session_state.page == "ℹ️ Home Page":
    import base64
    from PIL import Image

    # Logo base64 encode function
    def load_logo_base64(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    try:
        logo_base64 = load_logo_base64("BEstate.png")
        st.markdown(f"""
            <style>
                .logo-container {{
                    text-align: center;
                    margin-top: 10px;
                    margin-bottom: 10px;
                }}
                .logo-container img {{
                    width: 200px;
                }}
                .divider-line {{
                    border-top: 1px solid #ccc;
                    margin: 10px auto 20px auto;
                    width: 60%;
                }}
                .main-title {{
                    text-align: center;
                    font-size: 36px;
                    font-weight: bold;
                    margin-bottom: 30px;
                    color: #2c3e50;
                }}
                .nav-button {{
                    display: inline-block;
                    padding: 12px 20px;
                    margin: 10px;
                    background-color: #f2f2f2;
                    color: #333;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                    text-align: center;
                    font-weight: bold;
                    cursor: pointer;
                }}
                .nav-button:hover {{
                    background-color: #e0e0e0;
                }}
            </style>
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_base64}" alt="BEstate Logo">
            </div>
            <div class="divider-line"></div>
            <div class="main-title">Home Page</div>
        """, unsafe_allow_html=True)
    except:
        st.warning("Logo not found. Please ensure 'BEstate.png' is available in the folder.")

    # Dual introduction images
    try:
        with open("tasarım.png", "rb") as f1, open("evsahibikiracı.png", "rb") as f2:
            img1 = base64.b64encode(f1.read()).decode()
            img2 = base64.b64encode(f2.read()).decode()

        st.markdown(f"""
            <style>
            .dual-image-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 40px;
                margin-bottom: 30px;
            }}
            .dual-image-container img {{
                width: 420px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            </style>
            <div class="dual-image-container">
                <img src="data:image/png;base64,{img1}" alt="Istanbul Project Image">
                <img src="data:image/png;base64,{img2}" alt="Tenant vs Landlord Issue">
            </div>
        """, unsafe_allow_html=True)
    except:
        st.warning("One or both of the images could not be loaded. Please make sure 'tasarım.png' and 'evsahibikiracı.png' exist in the folder.")

    # Introduction text
    st.markdown("""
    ### 🌍 Who Are We?
We are **BEstate**, a platform aiming to make the **rental and sale housing market in Istanbul** more transparent, fair, and data-driven.  
We analyze real estate listings to provide accurate and reliable price predictions for tenants, landlords, buyers, and real estate agents.

### ❓ Problem Statement
The real estate market in Istanbul is highly volatile and subjective.  
Prices are often determined based on perception rather than data,  
which leads to uncertainty in both rental and purchase processes.

### 🎯 Our Objective
The main goal of this project is to develop **machine learning models**  
that can predict **rental and sale prices** in Istanbul by analyzing factors such as location, size, building condition, and mortgage eligibility.

### 🧠 How We Do It?
- We collected data from **hepsiemlak.com** using Selenium web scraping.  
- We cleaned and preprocessed the data for modeling.  
- For each price type (rental and sale), we tested **5 different models** and applied hyperparameter optimization.  
- We selected the best-performing **LightGBM models** for both cases.  
- We built a user-friendly interface using Streamlit:

    1. 🏠 Rent Prediction Page  
    2. 🏡 Sale Prediction Page  
    3. 📈 Data Visualization Page  
    4. ℹ️ Home Page  
    5. 🗺️ Heatmap Page  
    """)

    # Navigation buttons
    st.markdown("### 🧭 Explore Other Pages")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🏠 Rent Prediction"):
            st.session_state.page = "🏠 Rent Prediction"
            st.rerun()

    with col2:
        if st.button("📈 Data Visualization"):
            st.session_state.page = "📈 Data Visualization"
            st.rerun()

    with col3:
        if st.button("🗺️ Heatmap"):
            st.session_state.page = "🗺️ Heatmap"
            st.rerun()

    with col4:
        if st.button("🏡 Sale Prediction"):
            st.session_state.page = "🏡 Sale Prediction"
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 14px; color: gray; margin-top: 20px;'>
        📬 Contact us: <a href="mailto:destek@bestate.com">destek@bestate.com</a><br><br>
        © 2025 BEstate | All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# Page 5 - Heatmap
# --------------------------------
elif st.session_state.page == "🗺️ Heatmap":
    import json
    import folium
    from streamlit_folium import st_folium

    st.header("📍 Average Price Heatmap of Istanbul")

    # 1️⃣ Select Rent or Sale
    veri_turu = st.radio("Select Data Type", ["Rent", "Sale"], horizontal=True)

    if veri_turu == "Rent":
        df_selected = pd.read_csv("hepsiemlakpreprocessing.csv", encoding="utf-8")
        fiyat_sutun_adi = "Fiyat"
        fiyat_gosterim = "Average Rent (TL)"
    else:
        df_selected = pd.read_csv("satılıkpreprocessing.csv", encoding="utf-8")
        fiyat_sutun_adi = "Fiyat"
        fiyat_gosterim = "Average Sale (TL)"

    # 2️⃣ Normalization Function
    def normalize_name(name):
        name = str(name).lower().strip()
        name = name.replace("mah.", "").replace("mahallesi", "").replace("-", " ").replace("_", " ")
        replacements = {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"}
        for turk, eng in replacements.items():
            name = name.replace(turk, eng)
        return name

    harita_turu = st.selectbox("Select Map Type", ["By District", "By Neighborhood"])

    # -----------------------------
    # District-Level Heatmap
    # -----------------------------
    if harita_turu == "By District":
        avg_prices = df_selected.groupby("İlçe")[fiyat_sutun_adi].mean().reset_index()
        avg_prices.columns = ["Ilce", "Ortalama_Fiyat"]
        avg_prices["Ilce_normalized"] = avg_prices["Ilce"].apply(normalize_name)

        with open("ilce_geojson.json", "r", encoding="utf-8") as f:
            ilce_geojson = json.load(f)

        for feature in ilce_geojson["features"]:
            ilce_adi = feature["properties"]["display_name"].split(",")[0].strip()
            feature["properties"]["name"] = normalize_name(ilce_adi)

        fiyat_dict = dict(zip(avg_prices["Ilce_normalized"], avg_prices["Ortalama_Fiyat"]))
        for feature in ilce_geojson["features"]:
            ilce_adi = feature["properties"]["name"]
            feature["properties"]["avg_rent"] = round(fiyat_dict.get(ilce_adi, 0), 2)

        tum_ilceler = sorted(avg_prices["Ilce"].unique())
        secilen_ilce = st.selectbox("Select District:", ["All Districts"] + list(tum_ilceler))

        if secilen_ilce != "All Districts":
            secilen_ilce_norm = normalize_name(secilen_ilce)
            filtreli_geojson = {
                "type": "FeatureCollection",
                "features": [f for f in ilce_geojson["features"] if f["properties"]["name"] == secilen_ilce_norm]
            }
            tablo = avg_prices[avg_prices["Ilce"] == secilen_ilce]
        else:
            filtreli_geojson = ilce_geojson
            tablo = avg_prices.copy()

        m = folium.Map(location=[41.0082, 28.9784], zoom_start=10)

        folium.Choropleth(
            geo_data=filtreli_geojson,
            data=avg_prices,
            columns=["Ilce_normalized", "Ortalama_Fiyat"],
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            line_color='white',
            legend_name=fiyat_gosterim
        ).add_to(m)

        folium.GeoJson(
            filtreli_geojson,
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "avg_rent"],
                aliases=["District:", fiyat_gosterim + ":"],
                localize=True,
                sticky=False,
                labels=True
            )
        ).add_to(m)

        st_folium(m, width=None, height=800, use_container_width=True)
        st.subheader(f"📊 {fiyat_gosterim} for Selected Districts")
        st.dataframe(tablo.reset_index(drop=True))

    # -----------------------------
    # Neighborhood-Level Heatmap
    # -----------------------------
    elif harita_turu == "By Neighborhood":
        df_selected["Mahalle_Normalized"] = df_selected["Mahalle"].apply(normalize_name)
        df_selected["Ilce_Normalized"] = df_selected["İlçe"].apply(normalize_name)
        df_selected["key"] = df_selected["Mahalle_Normalized"] + " | " + df_selected["Ilce_Normalized"]

        mahalle_avg = df_selected.groupby("key").agg({
            fiyat_sutun_adi: ["mean", "count"],
            "Mahalle_Normalized": "first",
            "Ilce_Normalized": "first"
        }).reset_index()

        mahalle_avg.columns = ["key", "Average_Price", "Listing_Count", "Neighborhood", "District"]
        fiyat_dict = dict(zip(mahalle_avg["key"], mahalle_avg["Average_Price"]))
        ilan_dict = dict(zip(mahalle_avg["key"], mahalle_avg["Listing_Count"]))

        with open("mahalle_geojson.json", "r", encoding="utf-8") as f:
            mahalle_geojson = json.load(f)

        for feature in mahalle_geojson["features"]:
            props = feature["properties"]
            address = props.get("address", {})
            mahalle_adi = address.get("suburb") or address.get("village") or address.get("city") or ""
            ilce_adi = address.get("town") or address.get("city") or ""
            props["name"] = normalize_name(mahalle_adi)
            props["ilce"] = normalize_name(ilce_adi)
            props["key"] = props["name"] + " | " + props["ilce"]

        tum_ilceler = sorted(set([f["properties"]["ilce"].title() for f in mahalle_geojson["features"]]))
        secilen_ilce = st.selectbox("Select District (optional):", ["All Districts"] + tum_ilceler)

        if secilen_ilce != "All Districts":
            secilen_ilce_norm = normalize_name(secilen_ilce)
            mahalleler_filtreli = sorted(set(
                [f["properties"]["name"] for f in mahalle_geojson["features"]
                 if f["properties"]["ilce"] == secilen_ilce_norm and f["properties"]["key"] in fiyat_dict]
            ))
        else:
            mahalleler_filtreli = sorted(set(
                [f["properties"]["name"] for f in mahalle_geojson["features"]
                 if f["properties"]["key"] in fiyat_dict]
            ))

        secilen_mahalle = st.selectbox("Select Neighborhood:", ["All Neighborhoods"] + mahalleler_filtreli)

        for feature in mahalle_geojson["features"]:
            key = feature["properties"]["key"]
            feature["properties"]["avg_rent"] = round(fiyat_dict.get(key, 0), 2)
            feature["properties"]["ilan_count"] = ilan_dict.get(key, 0)

        if secilen_mahalle != "All Neighborhoods":
            filtreli_geojson = {
                "type": "FeatureCollection",
                "features": [f for f in mahalle_geojson["features"] if f["properties"]["name"] == secilen_mahalle]
            }
            tablo = mahalle_avg[mahalle_avg["Neighborhood"] == secilen_mahalle]
        elif secilen_ilce != "All Districts":
            filtreli_geojson = {
                "type": "FeatureCollection",
                "features": [f for f in mahalle_geojson["features"] if f["properties"]["ilce"] == secilen_ilce.lower()]
            }
            mahalleler = [f["properties"]["key"] for f in filtreli_geojson["features"]]
            tablo = mahalle_avg[mahalle_avg["key"].isin(mahalleler)]
        else:
            filtreli_geojson = mahalle_geojson
            tablo = mahalle_avg.copy()

        m = folium.Map(location=[41.0082, 28.9784], zoom_start=11)

        folium.Choropleth(
            geo_data=filtreli_geojson,
            data=mahalle_avg,
            columns=["key", "Average_Price"],
            key_on="feature.properties.key",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.4,
            legend_name=fiyat_gosterim
        ).add_to(m)

        for feature in filtreli_geojson["features"]:
            props = feature["properties"]
            name = props.get("name", "")
            ilce = props.get("ilce", "").title()
            kira = props.get("avg_rent", 0)
            ilan = props.get("ilan_count", 0)

            popup_text = f"""
            <b>Neighborhood:</b> {name.title()}<br>
            <b>District:</b> {ilce}<br>
            <b>{fiyat_gosterim}:</b> {kira:,.0f} TL<br>
            <b>Number of Listings:</b> {ilan}
            """

            folium.GeoJson(
                {"type": "FeatureCollection", "features": [feature]},
                tooltip=folium.GeoJsonTooltip(fields=["name"]),
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

        st_folium(m, width=None, height=800, use_container_width=True)
        st.subheader(f"📊 Average {fiyat_gosterim} for Selected Neighborhoods")
        st.dataframe(tablo.reset_index(drop=True))
