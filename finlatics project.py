# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:16:48 2025

@author: varsha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import pearsonr
df = pd.read_csv('Global Youtube Statistics.csv',encoding='unicode_escape')
unique_counts=df.nunique()
print(unique_counts)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


#The top 10 YouTube channels based on the number of subscribers
top_channels = df.sort_values(by='subscribers', ascending=False)
top_10_channels = top_channels.head(10)
print(top_10_channels[['Youtuber', 'subscribers']])


#the category with the highest average number of subscribers
category_avg_subscribers = df.groupby('category')['subscribers'].mean()
top_category = category_avg_subscribers.idxmax()
top_category_avg = category_avg_subscribers.max()
print(f"The category with the highest average number of subscribers is '{top_category}' with an average of {top_category_avg} subscribers.")

category_avg_videos = df.groupby('category')['uploads'].mean()
print(category_avg_videos)


#What are the top 5 countries with the highest number of YouTube channels?

country_channel_count = df.groupby('Country')['Youtuber'].count()
top_5_countries = country_channel_count.sort_values(ascending=False).head(5)
print(top_5_countries)


#What is the distribution of channel types across different categories?

category_channel_type_distribution = df.groupby(['category', 'channel_type']).size().unstack(fill_value=0)
print(category_channel_type_distribution)


#Is there a correlation between the number of subscribers and total video views for YouTube channels?
correlation = df['subscribers'].corr(df['video views'])

print(f"The correlation between the number of subscribers and total video views is: {correlation}")
plt.scatter(df['subscribers'], df['video views'])
plt.title('Subscribers vs Total Views')
plt.xlabel('Subscribers Count')
plt.ylabel('Total Views')
plt.show()
sns.regplot(x='subscribers', y='video views', data=df, scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.title('Regression Plot: Subscribers vs Total Views')
plt.xlabel('Subscribers Count')
plt.ylabel('Total Views')
plt.show()


#How do the monthly earnings vary throughout different categories?


df['earning_range'] = df['highest_monthly_earnings'] - df['lowest_monthly_earnings']##
category_avg_earning_range = df.groupby('category')['earning_range'].mean().sort_values(ascending=False)
print(category_avg_earning_range)
plt.figure(figsize=(10, 6))
sns.barplot(x=category_avg_earning_range.index, y=category_avg_earning_range.values)
plt.title('Average Earning Range per Category')
plt.xlabel('Category')
plt.ylabel('Average Earning Range')
plt.xticks(rotation=45)
plt.show()

#===========================================================================

df['change_in_subscribers'] = df['subscribers'].diff()
print("Subscriber Trend Over 30 Days:")
print(df[['subscribers', 'change_in_subscribers']])


plt.figure(figsize=(10, 6))
plt.plot(df.index + 1, df['subscribers'], marker='o', color='b', label='Total Subscribers Gained')
plt.title('Overall Trend in Subscribers Gained (Last 30 Days)')
plt.xlabel('Day')
plt.ylabel('Subscribers')
plt.xticks(range(1, 31))  # Marking the x-axis with days 1 to 30
plt.grid(True)
plt.legend()
plt.show()
#==========================================================================

#Are there any outliers in terms of yearly earnings from YouTube channels?
df['earnings_range'] = df['highest_yearly_earnings'] - df['lowest_yearly_earnings']
Q1_highest = df['highest_yearly_earnings'].quantile(0.25)
Q3_highest = df['highest_yearly_earnings'].quantile(0.75)
IQR_highest = Q3_highest - Q1_highest
lower_bound_highest = Q1_highest - 1.5 * IQR_highest
upper_bound_highest = Q3_highest + 1.5 * IQR_highest

outliers_highest = df[(df['highest_yearly_earnings'] < lower_bound_highest) | 
                     (df['highest_yearly_earnings'] > upper_bound_highest)]


Q1_lowest = df['lowest_yearly_earnings'].quantile(0.25)
Q3_lowest = df['lowest_yearly_earnings'].quantile(0.75)
IQR_lowest = Q3_lowest - Q1_lowest
lower_bound_lowest = Q1_lowest - 1.5 * IQR_lowest
upper_bound_lowest = Q3_lowest + 1.5 * IQR_lowest


outliers_lowest = df[(df['lowest_yearly_earnings'] < lower_bound_lowest) | 
                     (df['lowest_yearly_earnings'] > upper_bound_lowest)]
print("Outliers based on highest yearly earnings:")
print(outliers_highest)
print("Outliers based on lowest yearly earnings:")
print(outliers_lowest)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.boxplot(df['highest_yearly_earnings'], vert=False)
plt.title('Boxplot of Highest Yearly Earnings')
plt.xlabel('Highest Yearly Earnings')

plt.subplot(1, 2, 2)
plt.boxplot(df['lowest_yearly_earnings'], vert=False)
plt.title('Boxplot of Lowest Yearly Earnings')
plt.xlabel('Lowest Yearly Earnings')

plt.tight_layout()
plt.show()
#=========================================================================
plt.figure(figsize=(10, 6))
df['created_year'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Channel Creation Dates by Year')
plt.xlabel('Year of Creation')
plt.ylabel('Number of Channels')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10, 6))
df.groupby(['created_year', 'created_month']).size().unstack().plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20')
plt.title('Monthly Distribution of Channel Creation Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Channels')
plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df.groupby('created_year').size().plot(kind='line', marker='o', color='blue')
plt.title('Trend of Channel Creation Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Channels Created')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
#=============================================================================
channels_per_country = df.groupby('Country')['Youtuber'].nunique().reset_index()
channels_per_country = channels_per_country.rename(columns={'Youtuber': 'Number_of_YouTube_channels'})
df_merged = pd.merge(df, channels_per_country, on='Country', how='inner')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_merged, x='Gross tertiary education enrollment (%)', y='Number_of_YouTube_channels')
plt.title('Relationship between Gross Tertiary Education Enrollment and YouTube Channels')
plt.xlabel('Gross Tertiary Education Enrollment')
plt.ylabel('Number of YouTube Channels')
plt.show()
print("Summary Statistics:")
print(df_merged[['Gross tertiary education enrollment (%)', 'Number_of_YouTube_channels']].describe())

correlation = df_merged['Gross tertiary education enrollment (%)'].corr(df_merged['Number_of_YouTube_channels'])
print(f"\nCorrelation between Gross Tertiary Education Enrollment and Number of YouTube Channels: {correlation:.2f}")
#=============================================================================

channels_per_country = df.groupby('Country')['Youtuber'].nunique().reset_index()
channels_per_country = channels_per_country.rename(columns={'Youtuber': 'Number_of_YouTube_channels'})

top_10_countries = channels_per_country.sort_values('Number_of_YouTube_channels', ascending=False).head(10)

df_top_10 = pd.merge(top_10_countries, df[['Country', 'Unemployment rate']], on='Country', how='inner')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_top_10, x='Country', y='Unemployment rate', palette='viridis')
plt.title('Unemployment rate for the Top 10 Countries with the Highest Number of YouTube Channels')
plt.xlabel('Country')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Unemployment Rates for Top 10 Countries with Highest Number of YouTube Channels:")
print(df_top_10[['Country', 'Number_of_YouTube_channels', 'Unemployment rate']])
#============================================================================================
channels_per_country = df.groupby('Country')['Youtuber'].nunique().reset_index()
channels_per_country = channels_per_country.rename(columns={'Youtuber': 'Number_of_YouTube_channels'})


df_urban_population = df[['Country', 'Urban_population']].drop_duplicates()

df_merged = pd.merge(channels_per_country, df_urban_population, on='Country', how='inner')

average_urban_population_percentage = df_merged['Urban_population'].mean()

print(f"The average urban population percentage in countries with YouTube channels is: {average_urban_population_percentage:.2f}%")
plt.figure(figsize=(12, 6))
sns.barplot(data=df_merged, x='Country', y='Urban_population', palette='viridis')
plt.title('Urban Population Percentage in Countries with YouTube Channels')
plt.xlabel('Country')
plt.ylabel('Urban Population Percentage (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#===================================================================================

channels_per_country = df.groupby('Country')['Youtuber'].nunique().reset_index()
channels_per_country = channels_per_country.rename(columns={'Youtuber': 'Number_of_YouTube_channels'})

df_geo = pd.merge(channels_per_country, df[['Country', 'Latitude', 'Longitude']], on='Country', how='inner')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_geo, x='Longitude', y='Latitude', size='Number_of_YouTube_channels', hue='Number_of_YouTube_channels', palette='viridis', sizes=(20, 200))
plt.title('Distribution of YouTube Channels Based on Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Number of YouTube Channels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_geo, x='Longitude', y='Latitude', cmap='Blues', fill=True, thresh=0)
#====================================================================================

correlation = df[['subscribers', 'Population']].corr().iloc[0, 1]

print(f"The correlation between the number of YouTube subscribers and the population of a country is: {correlation:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Population', y='subscribers')
plt.title('Correlation Between YouTube Subscribers and Population of a Country')
plt.xlabel('Population')
plt.ylabel('Number of Subscribers')
plt.show()
#################################################
top_10_countries = df[['Country', 'Youtuber', 'Population']].sort_values(by='Youtuber', ascending=False).head(10)

fig, ax1 = plt.subplots(figsize=(12, 6))

sns.barplot(x='Country', y='Youtuber', data=top_10_countries, color='blue', ax=ax1)

ax1.set_title('Top 10 Countries with the Highest Number of YouTube Channels')
ax1.set_xlabel('Country')
ax1.set_ylabel('Number of YouTube Channels')
ax2 = ax1.twinx()  
sns.lineplot(x='Country', y='Population', data=top_10_countries, color='red', marker='o', ax=ax2)

ax2.set_ylabel('Population')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
###############################################################################

df_clean = df.dropna(subset=['subscribers_for_last_30_days', 'Unemployment rate'])

correlation, p_value = pearsonr(df_clean['subscribers_for_last_30_days'], df_clean['Unemployment rate'])

print(f'Pearson correlation coefficient: {correlation}')
print(f'P-value: {p_value}')


plt.figure(figsize=(10, 6))
sns.scatterplot(x='subscribers_for_last_30_days', y='Unemployment rate', data=df_clean)
plt.title('Correlation between Subscribers Gained in Last 30 Days and Unemployment Rate')
plt.xlabel('Subscribers Gained in Last 30 Days')
plt.ylabel('Unemployment Rate (%)')

plt.show()
#======================================================================

df_clean = df.dropna(subset=['video_views_for_the_last_30_days', 'channel_type'])

plt.figure(figsize=(12, 6))

sns.boxplot(x='channel_type', y='video_views_for_the_last_30_days', data=df_clean)

plt.title('Distribution of Video Views in the Last 30 Days Across Channel Types')
plt.xlabel('Channel Type')
plt.ylabel('Video Views in Last 30 Days')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
#===========================================================================
df_clean = df.dropna(subset=['uploads', 'video views'])
plt.figure(figsize=(10, 6))
sns.scatterplot(x='uploads', y='video views', data=df_clean)

plt.title('Relationship between Total Videos Uploaded and Video Views')
plt.xlabel('Total Videos Uploaded')
plt.ylabel('Video Views')

plt.tight_layout()
plt.show()

###################################################################################

current_date = datetime.now()
df['months_active'] = (current_date.year - df['created_year']) * 12 + current_date.month
df['subscribers_gained_since_creation'] = df['subscribers'] - df['subscribers_for_last_30_days']
df['avg_subscribers_per_month'] = df['subscribers_gained_since_creation'] / df['months_active']
average_subscribers_per_month = df['avg_subscribers_per_month'].mean()
print(f"The average number of subscribers gained per month is: {average_subscribers_per_month:.2f}")
