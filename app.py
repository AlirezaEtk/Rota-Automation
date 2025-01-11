import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import random


def load_data(file):
    data = pd.read_excel(file)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    return data

def train_forecast_model(data):
    valid_months = [12, 1, 2, 3]
    data = data[data['Month'].isin(valid_months)]
    features = ['Holiday Event', 'Delivery', 'for here']
    targets = ['Breakfast', 'Lunch', 'Dinner']
    results = {}

    X = data[features]

    for target in targets:
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[target] = {'Model': model, 'RMSE': rmse}
    return results

def forecast_sales(models, future_data):
    future_data['Day_of_Week'] = future_data['Date'].dt.dayofweek
    future_data['Month'] = future_data['Date'].dt.month

    future_data['Holiday Event'] = 0
    future_data['Delivery'] = 50  
    future_data['for here'] = 30  

    closed_days = [pd.Timestamp('2024-12-25'), pd.Timestamp('2025-01-01')]
    future_data.loc[future_data['Date'].isin(closed_days), ['Holiday Event', 'Delivery', 'for here']] = 0

    features = ['Holiday Event', 'Delivery', 'for here']
    for target, info in models.items():
        future_data[target] = info['Model'].predict(future_data[features])
        future_data.loc[future_data['Date'].isin(closed_days), target] = 0
 
    return future_data

def create_rota(forecast_data, unavailable_staff):
    staff = {
        'Manager': ['M1'],
        'Assistant Manager': ['ASM1'],
        'Supervisor': [f'S{i+1}' for i in range(4)],
        'Barista': [f'B{i+1}' for i in range(9)]
    }

    staff_hours = {role: {name: 0 for name in names} for role, names in staff.items()}

    def assign_staff(role, hours, current_date):
        available_staff = [
            name for name in staff[role]
            if not any(
                u['Staff'] == name and pd.Timestamp(u['Start Date']) <= current_date <= pd.Timestamp(u['End Date'])
                for u in st.session_state.unavailable_staff
            ) and staff_hours[role][name] + hours <= 40
        ]
        if available_staff:
            selected_staff = random.choice(available_staff)
            staff_hours[role][selected_staff] += hours
            return selected_staff
        return None

    rota_schedule = []
    for _, row in forecast_data.iterrows():
        date = pd.Timestamp(row['Date'])
        day_of_week = date.dayofweek

        if day_of_week == 6:  
            shifts = ['Day']
            shift_times = ['9:30 - 18:00']
            sales = [row['Breakfast'] + row['Lunch'] + row['Dinner']]
            base_staff = [6] 
        else: 
            shifts = ['Morning', 'Mid', 'Evening']
            shift_times = ['7:30 - 12:00', '12:00 - 17:00', '17:00 - 21:00']
            sales = [row['Breakfast'], row['Lunch'], row['Dinner']]
            base_staff = [3, 5, 3]

        for shift, shift_time, shift_sales, shift_base in zip(shifts, shift_times, sales, base_staff):
            num_staff = int(shift_base + shift_sales * 0.001)
            shift_data = {'Date': [], 'Shift': [], 'Shift Time': [], 'Role': [], 'Staff': []}

            if shift == 'Morning' or shift == 'Day':
                shift_data['Date'].append(date)
                shift_data['Shift'].append(shift)
                shift_data['Shift Time'].append(shift_time)
                shift_data['Role'].append('Manager')
                shift_data['Staff'].append(assign_staff('Manager', 8, date))
                for _ in range(num_staff - 1):
                    shift_data['Date'].append(date)
                    shift_data['Shift'].append(shift)
                    shift_data['Shift Time'].append(shift_time)
                    shift_data['Role'].append('Barista')
                    shift_data['Staff'].append(assign_staff('Barista', 8, date))
            elif shift == 'Mid':
                shift_data['Date'].append(date)
                shift_data['Shift'].append(shift)
                shift_data['Shift Time'].append(shift_time)
                shift_data['Role'].append('Assistant Manager')
                shift_data['Staff'].append(assign_staff('Assistant Manager', 8, date))
                for _ in range(num_staff - 1):
                    shift_data['Date'].append(date)
                    shift_data['Shift'].append(shift)
                    shift_data['Shift Time'].append(shift_time)
                    shift_data['Role'].append('Barista')
                    shift_data['Staff'].append(assign_staff('Barista', 8, date))
            else:  
                shift_data['Date'].append(date)
                shift_data['Shift'].append(shift)
                shift_data['Shift Time'].append(shift_time)
                shift_data['Role'].append('Supervisor')
                shift_data['Staff'].append(assign_staff('Supervisor', 8, date))
                for _ in range(num_staff - 1):
                    shift_data['Date'].append(date)
                    shift_data['Shift'].append(shift)
                    shift_data['Shift Time'].append(shift_time)
                    shift_data['Role'].append('Barista')
                    shift_data['Staff'].append(assign_staff('Barista', 8, date))

            rota_schedule.append(pd.DataFrame(shift_data))
    return pd.concat(rota_schedule, ignore_index=True)

st.title("Coffee Shop Sales Forecasting and Rota Automation")

if "models" not in st.session_state:
    st.session_state.models = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "unavailable_staff" not in st.session_state:
    st.session_state.unavailable_staff = []
if "start_date" not in st.session_state:
    st.session_state.start_date = None
if "end_date" not in st.session_state:
    st.session_state.end_date = None

uploaded_file = st.file_uploader("Upload historical sales data (Excel format):", type=["xlsx"])

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Historical Data Preview:")
    st.dataframe(data.head())

    st.session_state.start_date = st.date_input("Select start date:")
    st.session_state.end_date = st.date_input("Select end date:")

    unavailable_staff_input = st.text_input("Enter staff (e.g., B1, S2):")
    unavailability_start_date = st.date_input("Unavailable Start Date:")
    unavailability_end_date = st.date_input("Unavailable End Date:")

    if st.button("Add Unavailability"):
        st.session_state.unavailable_staff.append({
            "Staff": unavailable_staff_input,
            "Start Date": unavailability_start_date,
            "End Date": unavailability_end_date
        })
        st.success(f"Unavailability added for {unavailable_staff_input}")

    st.write("Unavailable Staff:")
    st.dataframe(pd.DataFrame(st.session_state.unavailable_staff))

    if st.button("Train Forecasting Model"):
        st.session_state.models = train_forecast_model(data)
        st.success("Models trained successfully!")

    
        future_dates = pd.date_range(start=st.session_state.start_date, end=st.session_state.end_date, freq='D')
        future_data = pd.DataFrame({'Date': future_dates})
        st.session_state.forecast = forecast_sales(st.session_state.models, future_data)

        st.write("Forecast Results:")
        st.dataframe(st.session_state.forecast)

    if st.session_state.models and st.session_state.forecast is not None:
        if st.button("Generate Rota"):
            rota = create_rota(st.session_state.forecast, st.session_state.unavailable_staff)
            st.write("Generated Rota:")
            st.dataframe(rota)

            
            rota_file = 'rota_schedule.csv'
            rota.to_csv(rota_file, index=False)
            st.download_button("Download Rota Schedule", data=open(rota_file).read(), file_name=rota_file)
