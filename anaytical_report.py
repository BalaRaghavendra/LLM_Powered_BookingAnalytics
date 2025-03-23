#analytical_report.py
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Read the data
df = pd.read_csv('hotel_bookings.csv')

# Data cleaning
df['children'] = df['children'].fillna(0).astype(int)
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + 
                                   df['arrival_date_month'] + '-' + 
                                   df['arrival_date_day_of_month'].astype(str))
df['revenue'] = df['adr'] * df['total_nights']

# 1. Revenue trends over time
def calculate_revenue_trends():
    actual_bookings = df[df['is_canceled'] == 0]
    monthly_revenue = actual_bookings.groupby(pd.Grouper(key='arrival_date', freq='ME'))['revenue'].sum().reset_index()
    monthly_revenue['month_year'] = monthly_revenue['arrival_date'].dt.strftime('%b %Y')
    
    hotel_revenue = actual_bookings.groupby('hotel')['revenue'].sum().reset_index()
    
    return {
        "monthly_revenue": monthly_revenue.to_dict(orient='records'),
        "hotel_revenue": hotel_revenue.to_dict(orient='records')
    }

# 2. Cancellation rate as percentage of total bookings
def calculate_cancellations():
    cancellation_rate = df['is_canceled'].mean() * 100
    
    hotel_cancellation = df.groupby('hotel')['is_canceled'].mean() * 100
    
    monthly_cancellation = df.groupby('arrival_date_month')['is_canceled'].mean() * 100
    
    df['lead_time_bucket'] = pd.cut(
        df['lead_time'], 
        bins=[0, 7, 30, 90, 180, 365, float('inf')],
        labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
    )
    lead_time_cancel = df.groupby('lead_time_bucket', observed=False)['is_canceled'].mean() * 100
    
    return {
        "overall_cancellation_rate": cancellation_rate,
        "hotel_cancellation": hotel_cancellation.to_dict(),
        "monthly_cancellation": monthly_cancellation.to_dict(),
        "lead_time_cancel": lead_time_cancel.to_dict()
    }

# 3. Geographical distribution of users
def calculate_geography():
    country_bookings = df['country'].value_counts().reset_index()
    country_bookings.columns = ['country', 'bookings']
    
    top10_countries = country_bookings.head(10)['country'].tolist()
    country_cancel = df[df['country'].isin(top10_countries)].groupby('country')['is_canceled'].mean() * 100
    
    return {
        "top_countries": country_bookings.head(15).to_dict(orient='records'),
        "country_cancel": country_cancel.to_dict()
    }

# 4. Booking Lead time distribution
def calculate_lead_time():
    df['lead_time_bucket'] = pd.cut(
        df['lead_time'], 
        bins=[0, 7, 30, 90, 180, 365, float('inf')],
        labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
    )
    lead_time_dist = df['lead_time_bucket'].value_counts().sort_index()
    
    avg_lead_by_segment = df.groupby('market_segment')['lead_time'].mean().sort_values(ascending=False)
    
    return {
        "lead_time_distribution": lead_time_dist.to_dict(),
        "avg_lead_by_segment": avg_lead_by_segment.to_dict()
    }

# 5. Additional Analytics
def calculate_additional_analytics():
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    avg_stay = df[df['is_canceled'] == 0].groupby('hotel')['total_nights'].mean()
    
    booking_changes = df.groupby('booking_changes')['is_canceled'].mean() * 100
    
    adr_by_month = df.groupby(['hotel', 'arrival_date_month'])['adr'].mean().unstack()
    
    df['got_assigned_same_room'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)
    room_match_rate = df.groupby('hotel')['got_assigned_same_room'].mean() * 100
    
    special_req_cancel = df.groupby('total_of_special_requests')['is_canceled'].mean() * 100
    
    return {
        "avg_length_of_stay": avg_stay.to_dict(),
        "cancellation_by_booking_changes": booking_changes.to_dict(),
        "adr_by_month": adr_by_month.to_dict(),
        "room_match_rate": room_match_rate.to_dict(),
        "cancellation_by_special_requests": special_req_cancel.to_dict()
    }

# Generate a comprehensive report with all analytics
def generate_report():
    # Run all calculations
    revenue_data = calculate_revenue_trends()
    cancel_data = calculate_cancellations()
    geo_data = calculate_geography()
    lead_time_data = calculate_lead_time()
    additional_data = calculate_additional_analytics()
    
    # Create a summary report as a dictionary
    report = {
        "revenue_analysis": revenue_data,
        "cancellation_analysis": cancel_data,
        "geographical_analysis": geo_data,
        "lead_time_analysis": lead_time_data,
        "additional_insights": additional_data
    }

    return report
    
    # Save the report as a JSON file
    

if __name__ == "__main__":
    report = generate_report()
    with open('hotel_bookings_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=4, default=str)  # Use default=str to handle Timestamp serialization
    
    print("Report generated successfully: hotel_bookings_analysis_report.json")
