import os

import pandas as pd
import phik
from dotenv import load_dotenv
from sqlalchemy import create_engine


class DataProcessor:
    def __init__(self, orders_path, events_path, currency_path):
        self.orders_path = orders_path
        self.events_path = events_path
        self.currency_path = currency_path
        self.df = None
        self.df_currency = None
        self.user_profile = None
        self.duplicates_count = 0

    def load_data(self):
        df_orders = pd.read_csv(self.orders_path)
        df_events = pd.read_csv(self.events_path)

        df = df_orders.merge(df_events, on='event_id', how='inner')
        df = df.rename(columns={
            'created_dt_msk': 'order_dt',
            'created_ts_msk': 'order_ts'
        })

        cols = [
            'user_id',
            'device_type_canonical',
            'order_id',
            'order_dt',
            'order_ts',
            'currency_code',
            'revenue',
            'tickets_count',
            'days_since_prev',
            'event_id',
            'service_name',
            'event_type_main',
            'region_name',
            'city_name'
        ]
        self.df = df[cols].copy()
        return self.df

    def load_data_from_db(self, query):
        load_dotenv()
        db_user = os.getenv('DB_USER')
        db_pwd = os.getenv('DB_PWD')
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')
        db_name = os.getenv('DB_NAME')

        engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}')
        self.df = pd.read_sql_query(query, con=engine)
        return self.df

    def load_currency(self):
        self.df_currency = pd.read_csv(self.currency_path)
        return self.df_currency

    def preprocess(self):
        key = ['user_id', 'order_ts', 'device_type_canonical', 'service_name', 'event_id', 'revenue']
        self.duplicates_count = self.df.duplicated(subset=key).sum()
        df = self.df.drop_duplicates(subset=key)

        df = df.merge(self.df_currency, left_on='order_dt', right_on='data', how='left')

        df['revenue_rub'] = df['revenue']
        mask_kzt = df['currency_code'] == 'kzt'
        df.loc[mask_kzt, 'revenue_rub'] = df.loc[mask_kzt, 'revenue'] * df.loc[mask_kzt, 'curs'] / 100
        df = df.drop(columns=['data', 'nominal', 'curs', 'cdx'])

        df['tickets_count'] = pd.to_numeric(df['tickets_count'], downcast='integer')
        df['days_since_prev'] = pd.to_numeric(df['days_since_prev'], downcast='float')
        df['order_dt'] = pd.to_datetime(df['order_dt'])
        df['order_ts'] = pd.to_datetime(df['order_ts'])

        df = df[df['revenue_rub'] > 0]
        quantile_99 = df['revenue_rub'].quantile(0.99)
        df = df[df['revenue_rub'] <= quantile_99].copy()

        self.df = df
        return self.df

    def build_user_profile(self):
        df = self.df.sort_values(by='order_ts')
        grouped = df.groupby('user_id')

        user_profile = pd.DataFrame()
        user_profile['date_first_order'] = grouped['order_ts'].first()
        user_profile['date_last_order'] = grouped['order_ts'].last()
        user_profile['device_first_order'] = grouped['device_type_canonical'].first()
        user_profile['region_first_order'] = grouped['region_name'].first()
        user_profile['event_first_order'] = grouped['event_type_main'].first()
        user_profile['total_orders'] = grouped['order_id'].nunique()
        user_profile['avg_revenue'] = grouped['revenue_rub'].mean()
        user_profile['avg_duration'] = grouped['days_since_prev'].mean()
        user_profile['is_two'] = user_profile['total_orders'] >= 2
        user_profile['first_order_weekday'] = user_profile['date_first_order'].dt.dayofweek

        quantile_95 = user_profile['total_orders'].quantile(0.95)
        user_profile = user_profile[user_profile['total_orders'] <= quantile_95].copy()

        self.user_profile = user_profile
        return self.user_profile

    def retention_by_segment(self, column):
        result = self.user_profile.groupby(column).agg(
            users=('is_two', 'count'),
            return_rate=('is_two', 'mean')
        )
        result = result.sort_values(by='users', ascending=False)
        return result

    def check_hypothesis_1(self):
        profiles = self.user_profile
        profiles = profiles[profiles['event_first_order'].isin(['спорт', 'концерты'])]

        result = profiles.groupby('event_first_order').agg(
            users=('is_two', 'count'),
            return_rate=('is_two', 'mean')
        )
        return result

    def check_hypothesis_2(self):
        region_stats = self.retention_by_segment('region_first_order')
        top_regions = region_stats.head(10).index

        profiles = self.user_profile.copy()
        profiles['region_group'] = profiles['region_first_order'].isin(top_regions)
        profiles['region_group'] = profiles['region_group'].map({True: 'топ-10 регионов', False: 'остальные регионы'})

        result = profiles.groupby('region_group').agg(
            users=('is_two', 'count'),
            return_rate=('is_two', 'mean')
        )

        region_big = region_stats[region_stats['users'] >= 50]
        corr = region_big['users'].corr(region_big['return_rate'], method='spearman')

        return result, corr

    def correlation_analysis(self):
        profiles = self.user_profile.copy()
        profiles['orders_segment'] = pd.cut(
            profiles['total_orders'],
            bins=[0, 1, 2, 4, float('inf')],
            labels=['1 заказ', '2 заказа', '3-4 заказа', '5+ заказов']
        )

        cols = [
            'device_first_order',
            'region_first_order',
            'event_first_order',
            'first_order_weekday',
            'avg_revenue',
            'avg_duration',
            'total_orders',
            'orders_segment'
        ]
        corr = profiles[cols].phik_matrix(interval_cols=['avg_revenue', 'avg_duration', 'total_orders'])

        result = corr[['total_orders', 'orders_segment']]
        result = result.drop(['total_orders', 'orders_segment'])
        return result
