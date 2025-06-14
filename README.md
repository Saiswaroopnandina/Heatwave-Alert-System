# Heatwave-Alert-System

    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import joblib
    
    def load_data(csv_file):
        df = pd.read_csv(csv_file, parse_dates=["Date"])
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        return df[["DayOfYear", "Year", "Month", "Day"]], df["Temperature"]
    
    def train_model(csv_file):
        X, y = load_data(csv_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(X_train, y_train)
    
        preds = model.predict(X_test)
        print("MAE:", mean_absolute_error(y_test, preds))
    
        joblib.dump(model, "xgb_heat_model.pkl")
        print("XGBoost model saved.")
    
    if __name__ == "__main__":
        train_model("temperature_data.csv")



    import streamlit as st
    import pandas as pd
    import joblib
    import numpy as np
    
    st.title("🔥 Summer Heatwave Alert System (XGBoost)")
    model = joblib.load("xgb_heat_model.pkl")
    
    uploaded_file = st.file_uploader("Upload temperature CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
    
        X = df[["DayOfYear", "Year", "Month", "Day"]]
        predictions = model.predict(X)
        df["Predicted_Temperature"] = predictions
    
        st.line_chart(df.set_index("Date")[["Predicted_Temperature"]])
    
        alerts = df[df["Predicted_Temperature"] >= 40]
        if not alerts.empty:
            st.error("⚠️ Heatwave Alert on:")
            st.dataframe(alerts[["Date", "Predicted_Temperature"]])
        else:
            st.success("✅ No heatwave forecasted.")

