import pandas as pd
import numpy as np

class SupplyChainSimulator:
    """
    A class to simulate supply chain operations by combining weather and
    product data to calculate demand and lead times.

    This class encapsulates all the logic for data processing, demand calculation,
    and lead time determination. It provides a clean, reusable interface for
    running a supply chain simulation based on various factors.

    Example usage:
    >>> # Assuming df_weather and df_products are pre-loaded pandas DataFrames
    >>> # with the required columns.
    >>> simulator = SupplyChainSimulator(df_weather, df_products)
    >>> df_combined = simulator.run_simulation()
    >>> print(df_combined.head())
    """
    
    def __init__(self, df_weather: pd.DataFrame, df_products: pd.DataFrame):
        self.df_weather = df_weather
        self.df_products = df_products
        # Use a fixed seed for reproducibility
        self.rng = np.random.default_rng(42)
        self.month_map = self._create_month_map()

    def _create_month_map(self):
        """Creates a sorted dictionary mapping month numbers to names."""
        month = self.df_weather['LPO'].dt.month.unique()
        month_name = self.df_weather['LPO'].dt.month_name().unique()
        zipped_months = zip(month, month_name)
        return dict(sorted(dict(zipped_months).items()))

    def _check_seasonality(self, seasonality_list, month_name):
        """Checks if a product is in its season."""
        if not seasonality_list or month_name in seasonality_list:
            return True
        return False

    def _calculate_lead_time(self, distance, weather_severity, day_classification):
        """Calculates realistic lead time considering multiple factors."""
        base_lead_time = max(1, round(distance / 100))
        
        # Adjust for weather severity
        if weather_severity == 'Catastrophic':
            base_lead_time += self.rng.integers(5, 7)
        elif weather_severity == 'Extreme':
            base_lead_time += self.rng.integers(3, 5)
        elif weather_severity == 'Severe':
            base_lead_time += self.rng.integers(2, 3)
        elif weather_severity == 'Moderate':
            base_lead_time += self.rng.integers(1, 2)
        
        # Adjust for weekend
        if day_classification in ['Saturday', 'Sunday']:
            base_lead_time += 1
        
        # Ensure lead time is within limits
        return max(1, min(21, base_lead_time))

    def _calculate_demand_factor(self, weather_row, product_row, current_month, seed):
        """Calculates demand factor based on multiple factors."""
        demand_factor = 1.0
        
        # Use a local RNG seeded with a unique value for each date-product pair
        local_rng = np.random.default_rng(hash((seed, product_row['product_id'])) % 1000)

        # Seasonality
        if self._check_seasonality(product_row['seasonality'], current_month):
            demand_factor *= local_rng.uniform(1.2, 1.6)
        else:
            demand_factor *= local_rng.uniform(0.6, 0.9)
        
        # Precipitation
        precip_class = weather_row['precipitation_classification']
        if precip_class == 'Violent Rainfall':
            demand_factor *= local_rng.uniform(0.5, 0.65)
        elif precip_class == 'Heavy Rain':
            demand_factor *= local_rng.uniform(0.7, 0.85)
        elif precip_class == 'Moderate Rain':
            demand_factor *= local_rng.uniform(0.85, 0.95)
        elif precip_class == 'Light Rain':
            demand_factor *= local_rng.uniform(0.95, 1.05)
        
        # Temperature
        temp_class = weather_row['temperature_classification']
        if temp_class == 'Very Hot':
            demand_factor *= local_rng.uniform(1.3, 1.6)
        elif temp_class == 'Hot':
            demand_factor *= local_rng.uniform(1.1, 1.4)
        elif temp_class == 'Warm':
            demand_factor *= local_rng.uniform(1.0, 1.2)
        elif temp_class == 'Mild to Temperate':
            demand_factor *= local_rng.uniform(0.9, 1.1)
        elif temp_class == 'Cool':
            demand_factor *= local_rng.uniform(0.8, 1.0)
        elif temp_class == 'Cold':
            demand_factor *= local_rng.uniform(0.6, 0.9)
        elif temp_class == 'Very Cold':
            demand_factor *= local_rng.uniform(0.4, 0.7)
        
        # Wind
        wind_class = weather_row['wind_classification']
        if wind_class == 'Storm / Hurricane Force':
            demand_factor *= local_rng.uniform(0.4, 0.6)
        elif wind_class == 'Very Strong Wind / Gale':
            demand_factor *= local_rng.uniform(0.6, 0.8)
        elif wind_class == 'Moderate to Strong Wind':
            demand_factor *= local_rng.uniform(0.8, 0.95)
        elif wind_class == 'Gentle to Fresh Breeze':
            demand_factor *= local_rng.uniform(0.95, 1.05)
        
        # Holidays and weekends
        if weather_row['is_holiday']:
            demand_factor *= local_rng.uniform(1.3, 1.8)
        elif weather_row['day_classification'] in ['Saturday', 'Sunday']:
            demand_factor *= local_rng.uniform(1.2, 1.5)
        
        # Overall weather severity
        severity = weather_row['weather_severity']
        if severity == 'Catastrophic':
            demand_factor *= local_rng.uniform(0.2, 0.4)
        elif severity == 'Extreme':
            demand_factor *= local_rng.uniform(0.3, 0.6)
        elif severity == 'Severe':
            demand_factor *= local_rng.uniform(0.5, 0.8)
        elif severity == 'Moderate':
            demand_factor *= local_rng.uniform(0.8, 1.0)
        
        # Security limits
        demand_factor = max(0.1, min(3.0, demand_factor))
        
        return round(demand_factor, 3)

    def run_simulation(self):
        """
        Runs the simulation to create a combined DataFrame of weather,
        product, demand, and lead time data.
        """
        combined_data = []
        
        for _, weather_row in self.df_weather.iterrows():
            current_date = weather_row['LPO']
            current_month = self.month_map.get(current_date.month, 'Unknown')

            for _, product_row in self.df_products.iterrows():
                is_in_season = self._check_seasonality(product_row['seasonality'], current_month)
                
                # Use the date's timestamp as the seed for consistent results
                demand_factor = self._calculate_demand_factor(
                    weather_row, 
                    product_row, 
                    current_month, 
                    current_date.timestamp()
                )
                
                lead_time = self._calculate_lead_time(
                    product_row['distance_km'],
                    weather_row['weather_severity'],
                    weather_row['day_classification']
                )
                
                base_demand = self.rng.integers(5, 20)
                adjusted_demand = round(base_demand * demand_factor)
                
                combined_record = {
                    'date': current_date,
                    'product_id': product_row['product_id'],
                    'product': product_row['product'],
                    'category': product_row['category'],
                    'sub_category': product_row['sub_category'],
                    'supplier_id': product_row['supplier_id'],
                    'supplier': product_row['supplier'],
                    'is_in_season': is_in_season,
                    'demand_factor': demand_factor,
                    'adjusted_demand': adjusted_demand,
                    'lead_time_days': lead_time,
                    'shelf_life_days': product_row['shelf_life_days'],
                    'min_stock': product_row['min_stock'],
                    'max_stock': product_row['max_stock'],
                    'reorder_point': product_row['reorder_point'],
                    'distance_km': product_row['distance_km'],
                    'supplier_rating': product_row['supplier_rating'],
                    'temperature': weather_row['daily_average_temperature_c'],
                    'precipitation': weather_row['daily_total_precipitation_mm'],
                    'wind_speed': weather_row['daily_average_wind_speed_mps'],
                    'weather_severity': weather_row['weather_severity'],
                    'is_weekend': weather_row['is_weekend'],
                    'is_holiday': weather_row['is_holiday']
                }
                combined_data.append(combined_record)
        
        return pd.DataFrame(combined_data)
    
    def create_balanced_delivery(self, max_products_per_day: int = 10, out_of_season_percentage: float = 0.3):
        """
            Creates a balanced delivery schedule by selecting products for delivery each day,
            considering seasonality, holidays, and weekends.
            
            This method generates a realistic delivery schedule where the number of products
            delivered each day varies based on whether it's a weekday, weekend, or holiday.
            It ensures a mix of in-season and out-of-season products while respecting
            supplier preferences and product availability.
            
            Parameters:
            -----------
            max_products_per_day : int, optional
                Maximum number of products that can be delivered on a normal business day.
                This is the upper limit for weekdays; weekends and holidays have reduced limits.
                Default is 10.
                
            out_of_season_percentage : float, optional
                Target percentage of out-of-season products to include in daily deliveries.
                Must be between 0.0 and 1.0. The actual percentage may vary based on
                product availability. Default is 0.3 (30%).
                
            Returns:
            --------
            tuple
                A tuple containing two pandas DataFrames:
                
                - df_final : pd.DataFrame
                    Contains the detailed delivery schedule with columns including:
                    ['date', 'product_id', 'product', 'category', 'sub_category', 
                    'supplier_id', 'supplier', 'is_in_season', 'demand_factor', 
                    'adjusted_demand', 'lead_time_days', 'shelf_life_days', 
                    'min_stock', 'max_stock', 'reorder_point', 'distance_km', 
                    'supplier_rating', 'temperature', 'precipitation', 'wind_speed', 
                    'weather_severity', 'is_weekend', 'is_holiday']
                    
                - df_daily_stats : pd.DataFrame
                    Contains daily delivery statistics with columns:
                    ['date', 'products_delivered', 'in_season_delivered', 
                    'out_of_season_delivered', 'is_holiday', 'is_weekend', 
                    'total_in_season_available', 'total_out_of_season_available']
            
            Notes:
            ------
            - Delivery quantities are reduced on weekends and holidays according to
            predefined probability distributions
            - Products are selected based on supplier rating (higher rated first) and
            distance (closer suppliers first)
            - The method ensures at least one out-of-season product is included when
            available and delivery quantity permits
            - Actual delivery numbers may be lower than max_products_per_day due to
            product availability constraints
            
            Example usage:
            --------------
            >>> # Basic usage with default parameters
            >>> deliveries, stats = simulator.create_balanced_delivery()
            >>> 
            >>> # Custom parameters: allow more products but fewer out-of-season
            >>> deliveries, stats = simulator.create_balanced_delivery(
            ...     max_products_per_day=15,
            ...     out_of_season_percentage=0.2
            ... )
            >>> 
            >>> # Focus only on in-season products
            >>> deliveries, stats = simulator.create_balanced_delivery(
            ...     out_of_season_percentage=0.0
            ... )
            >>> 
            >>> # Analyze results
            >>> print(f"Total deliveries: {len(deliveries)}")
            >>> print(f"Average daily deliveries: {stats['products_delivered'].mean():.1f}")
        """
        
        
        df_combined = self.run_simulation()
        # First ensure one supplier per product per day
        df_sorted = df_combined.sort_values(['product', 'date', 'supplier_rating', 'distance_km'], 
                                        ascending=[True, True, False, True])
        df_base = df_sorted.groupby(['product', 'date']).first().reset_index()
        
        # Separate in-season and out-of-season products
        df_in_season = df_base[df_base['is_in_season'] == True]
        df_out_of_season = df_base[df_base['is_in_season'] == False]
        
        final_data = []
        daily_stats = []
        
        for date in sorted(df_base['date'].unique()):
            date_data_in_season = df_in_season[df_in_season['date'] == date]
            date_data_out_of_season = df_out_of_season[df_out_of_season['date'] == date]
            
            # Check if it's a holiday or weekend
            sample_row = date_data_in_season.iloc[0] if len(date_data_in_season) > 0 else date_data_out_of_season.iloc[0] if len(date_data_out_of_season) > 0 else None
            
            if sample_row is None:
                continue
                
            is_holiday = sample_row['is_holiday']
            is_weekend = sample_row['is_weekend']
            
            # Define total number of products for the day
            if is_holiday:
                n_products = self.rng.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
            elif is_weekend:
                n_products = self.rng.choice([0, 1, 2, 3, 4, 5], p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
            else:
                n_products = self.rng.integers(3, max_products_per_day)
            
            # Ensure it doesn't exceed available products
            total_available = len(date_data_in_season) + len(date_data_out_of_season)
            n_products = min(n_products, total_available)
            
            if n_products == 0:
                # Day without deliveries
                selected_products = pd.DataFrame()
            else:
                # Calculate how many out-of-season products to include (minimum 1 if available)
                n_out_of_season = max(0, min(len(date_data_out_of_season), 
                                        int(n_products * out_of_season_percentage)))
                
                if n_out_of_season == 0 and len(date_data_out_of_season) > 0 and n_products > 1:
                    # Ensure at least 1 out-of-season product if available
                    n_out_of_season = 1
                
                n_in_season = n_products - n_out_of_season
                
                # Select in-season products (ordered by demand)
                if n_in_season > 0:
                    selected_in_season = date_data_in_season.head(n_in_season)
                else:
                    selected_in_season = pd.DataFrame()
                
                # Select out-of-season products (ordered by demand)
                if n_out_of_season > 0:
                    selected_out_of_season = date_data_out_of_season.head(n_out_of_season)
                else:
                    selected_out_of_season = pd.DataFrame()
                
                # Combine selections
                selected_products = pd.concat([selected_in_season, selected_out_of_season])
            
            final_data.extend(selected_products.to_dict('records'))
            
            # Record daily statistics
            daily_stats.append({
                'date': date,
                'products_delivered': n_products,
                'in_season_delivered': len(selected_in_season),
                'out_of_season_delivered': len(selected_out_of_season),
                'is_holiday': is_holiday,
                'is_weekend': is_weekend,
                'total_in_season_available': len(date_data_in_season),
                'total_out_of_season_available': len(date_data_out_of_season)
            })
        
        df_final = pd.DataFrame(final_data).reset_index(drop=True)
        df_daily_stats = pd.DataFrame(daily_stats)
        
        return df_final, df_daily_stats