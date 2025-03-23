#rag_preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime
import json
from anaytical_report import generate_report

# Read the data
df = pd.read_csv('hotel_bookings.csv')

# Data cleaning
df['children'] = df['children'].fillna(0).astype(int)
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + 
                                   df['arrival_date_month'] + '-' + 
                                   df['arrival_date_day_of_month'].astype(str))
df['revenue'] = df['adr'] * df['total_nights']

# 1. Overall Insights
def calculate_overall_insights():
    actual_bookings = df[df['is_canceled'] == 0]
    return {
        "total_revenue": actual_bookings['revenue'].sum(),
        "average_price": df['adr'].mean(),
        "cancellation_rate": df['is_canceled'].mean() * 100,
        "average_stay_length": actual_bookings['total_nights'].mean(),
        "total_bookings": df.shape[0]
    }

# 2. Hotel-specific Insights
def calculate_hotel_insights():
    actual_bookings = df[df['is_canceled'] == 0]
    return {
        "revenue_by_hotel": actual_bookings.groupby('hotel')['revenue'].sum().to_dict(),
        "cancellation_rate_by_hotel": df.groupby('hotel')['is_canceled'].mean().mul(100).to_dict(),
        "average_price_by_hotel": df.groupby('hotel')['adr'].mean().to_dict(),
        "average_stay_length_by_hotel": actual_bookings.groupby('hotel')['total_nights'].mean().to_dict()
    }

# 3. Country-specific Insights
def calculate_country_insights():
    actual_bookings = df[df['is_canceled'] == 0]
    return {
        "top_countries_by_bookings": df['country'].value_counts().head(15).to_dict(),
        "cancellation_rate_by_country": df.groupby('country')['is_canceled'].mean().mul(100).to_dict(),
        "average_price_by_country": df.groupby('country')['adr'].mean().to_dict(),
        "average_stay_length_by_country": actual_bookings.groupby('country')['total_nights'].mean().to_dict()
    }

# 4. Monthly Insights
def calculate_monthly_insights():
    actual_bookings = df[df['is_canceled'] == 0]

    # Convert Timestamp keys to string
    def convert_keys_to_str(data):
        return {k.strftime('%Y-%m'): v for k, v in data.items() if pd.notna(k)}

    monthly_data = actual_bookings.groupby(pd.Grouper(key='arrival_date', freq='ME'))
    
    return {
        "monthly_revenue": convert_keys_to_str(monthly_data['revenue'].sum().to_dict()),
        "monthly_cancellation_rate": convert_keys_to_str(df.groupby(pd.Grouper(key='arrival_date', freq='ME'))['is_canceled'].mean().mul(100).to_dict()),
        "monthly_average_price": convert_keys_to_str(df.groupby(pd.Grouper(key='arrival_date', freq='ME'))['adr'].mean().to_dict()),
        "monthly_bookings": convert_keys_to_str(df.groupby(pd.Grouper(key='arrival_date', freq='ME')).size().to_dict())
    }

# 5. Geographical Insights
def calculate_geographical_insights():
    actual_bookings = df[df['is_canceled'] == 0]
    return {
        "top_countries_by_revenue": actual_bookings.groupby('country')['revenue'].sum().nlargest(10).to_dict(),
        "top_countries_by_cancellations": df.groupby('country')['is_canceled'].sum().nlargest(10).to_dict(),
        "top_countries_by_bookings": df['country'].value_counts().head(10).to_dict()
    }

# 6. Generate Comprehensive Report
def generate_insights():
    return {
        "overall_insights": calculate_overall_insights(),
        "hotel_insights": calculate_hotel_insights(),
        "country_insights": calculate_country_insights(),
        "monthly_insights": calculate_monthly_insights(),
        "geographical_insights": calculate_geographical_insights()
    }

def generate_data():
    return generate_insights() | generate_report()

# Save the report as a JSON file
if __name__ == "__main__":
    rag_data = generate_data()
    with open('hotel_bookings_rag_data.json', 'w') as f:
        json.dump(rag_data, f, indent=4, default=str)  # Use default=str to handle Timestamp serialization
    print("Report generated successfully: hotel_bookings_analysis_report.json")
