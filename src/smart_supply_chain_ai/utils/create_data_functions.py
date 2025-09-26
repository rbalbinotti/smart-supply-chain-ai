import pandas as pd
import numpy as np
import ast
import holidays


def create_min_max_stock(
    df: pd.DataFrame,
    base_min: pd.Series,
    base_max: pd.Series,
    base_reorder: pd.Series
) -> pd.DataFrame:
    """
    Generates randomized minimum stock, maximum stock, and reorder point values for inventory items,
    and appends them as new columns to the input DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The original DataFrame containing inventory items.
    base_min : pd.Series
        Base minimum stock levels for each item.
    base_max : pd.Series
        Base maximum stock levels for each item.
    base_reorder : pd.Series
        Base reorder point levels for each item.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with three new columns:
        - 'min_stock': randomized minimum stock (clipped to at least 1)
        - 'max_stock': randomized maximum stock (at least one unit above min_stock)
        - 'reorder_point': randomized reorder point (between min_stock + 1 and max_stock - 1)

    Notes:
    -----
    - Randomness is seeded with 42 for reproducibility.
    - Random variation is applied in the range [-2, 1] to each base value.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'item': ['A', 'B', 'C']})
    >>> base_min = pd.Series([5, 10, 15])
    >>> base_max = pd.Series([20, 25, 30])
    >>> base_reorder = pd.Series([10, 15, 20])
    >>> create_min_max_stock(data, base_min, base_max, base_reorder)
       item  min_stock  max_stock  reorder_point
    0     A          4         20             10
    1     B          9         25             15
    2     C         14         30             20
    """

    rng = np.random.default_rng(seed=42)

    # Apply random variation to base values
    min_stock = base_min + rng.integers(-2, 2, size=len(base_min))
    max_stock = base_max + rng.integers(-2, 2, size=len(base_max))
    reorder_point = base_reorder + rng.integers(-2, 2, size=len(base_reorder))

    # Ensure logical constraints
    df['min_stock'] = min_stock.clip(lower=1)
    df['max_stock'] = np.maximum(min_stock + 1, max_stock)
    df['reorder_point'] = np.maximum(min_stock + 1, np.minimum(reorder_point, max_stock - 1))

    return df



def simulate_purchase_order_columns(df, random_state=None):
    """
    Simulates purchase order-related columns based on product attributes, logistics, and calendar effects.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the following columns:
        - 'received_date': date when goods were received
        - 'sales_volume': recent sales volume
        - 'min_stock': minimum stock threshold
        - 'shelf_life_days': shelf life of the product in days
        - 'sales_demand': demand level (e.g., 'High', 'Low', 'Normal')
        - 'distance_km': delivery distance in kilometers
        - 'weather_severity': weather impact ('Severe', 'Moderate', 'Mild')

    random_state : int, optional
        Seed for random number generation to ensure reproducibility.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with simulated purchase order columns:
        - 'order_date': date the order was placed

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'received_date': ['2025-09-20', '2025-09-21'],
    ...     'sales_volume': [10, 5],
    ...     'min_stock': [20, 15],
    ...     'shelf_life_days': [5, 60],
    ...     'sales_demand': ['High', 'Low'],
    ...     'distance_km': [800, 300],
    ...     'weather_severity': ['Moderate', 'Severe']
    ... })
    >>> simulate_purchase_order_columns(df, random_state=42)
           order_date
    0 2025-09-13
    1 2025-09-14
    """
    if random_state is not None:
        np.random.seed(random_state)

    result_df = pd.DataFrame(index=df.index)

    def process_order(row):
        received_date = pd.to_datetime(row['received_date'])

        # Order date: 2 to 10 days before received date
        days_before = np.random.randint(2, 11)
        order_date = received_date - pd.Timedelta(days=days_before)

        # Base quantity based on sales and minimum stock
        base_qty = max(row['sales_volume'] * 5, row['min_stock'] * 2)

        # Adjust for perishability
        if row['shelf_life_days'] <= 7:
            order_qty = base_qty * 0.7  # Smaller orders for highly perishable items
        elif row['shelf_life_days'] <= 30:
            order_qty = base_qty * 1.0
        else:
            order_qty = base_qty * 1.5  # Larger orders for non-perishables

        # Adjust for demand
        if 'High' in str(row['sales_demand']):
            order_qty *= 1.3
        elif 'Low' in str(row['sales_demand']):
            order_qty *= 0.7

        order_qty = int(round(max(row['min_stock'], order_qty)))

        # Base delay based on distance
        base_delay = max(1, row['distance_km'] // 400)

        # Weather impact on delay
        if row['weather_severity'] == 'Severe':
            base_delay += 3
        elif row['weather_severity'] == 'Moderate':
            base_delay += 1

        # Actual delay with variability
        real_delay = base_delay + np.random.randint(0, 3)

        estimated_delivery = order_date + pd.Timedelta(days=days_before)
        actual_delivery = order_date + pd.Timedelta(days=days_before + real_delay)
        delay_days = max(0, (actual_delivery - received_date).days)

        return pd.Series({
            'order_date': order_date,
            # Uncomment below to include more columns:
            # 'order_quantity': order_qty,
            # 'estimated_delivery_date': estimated_delivery,
            # 'actual_delivery_date': actual_delivery,
            # 'delivery_delay_days': delay_days
        })

    order_data = df.apply(process_order, axis=1)
    return order_data


def simulate_sales_volume(df, random_state=None):
    """
    Simulates expected sales volume for retail inventory items based on category-specific turnover targets
    and various influencing factors such as shelf life, demand level, seasonality, calendar effects, and weather.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing inventory data with the following required columns:
        - 'stock_quantity': Current stock quantity of the item.
        - 'min_stock': Minimum stock threshold.
        - 'category': Product category (e.g., 'Fresh Foods', 'Dairy', etc.).
        - 'shelf_life_days': Shelf life of the item in days.
        - 'sales_demand': Demand level ('Very High', 'High', 'Normal', 'Low').
        - 'in_season': Boolean indicating if the item is in season.
        - 'is_holiday': Boolean indicating if the current day is a holiday.
        - 'is_weekend': Boolean indicating if the current day is a weekend.
        - 'weather_severity': Weather condition ('Severe', 'Moderate', 'Mild').

    random_state : int, optional
        Seed for the random number generator to ensure reproducibility of simulated results.

    Returns
    -------
    pandas.Series
        A Series of simulated sales volumes (integers), one for each row in the input DataFrame.

    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'stock_quantity': [50, 20],
    ...     'min_stock': [10, 5],
    ...     'category': ['Fresh Foods', 'Pantry'],
    ...     'shelf_life_days': [1, 30],
    ...     'sales_demand': ['High', 'Normal'],
    ...     'in_season': [True, False],
    ...     'is_holiday': [False, True],
    ...     'is_weekend': [True, False],
    ...     'weather_severity': ['Mild', 'Moderate']
    ... })
    >>> simulate_sales_volume_controlled_turnover(data, random_state=42)
    0    251
    1     47
    dtype: int64
    """

    if random_state is not None:
        np.random.seed(random_state)

    def calculate_sales_per_row(row):
        stock_quantity = row['stock_quantity']
        min_stock = row['min_stock']

        # Category-specific base turnover targets
        base_turnover = {
            'Oils & Condiments': 1.1,
            'Grains & Flours': 1.3,
            'Breads & Biscuits': 3.0,
            'Fresh Foods': 2.5,      # Very high - multiple restocks per day
            'Dairy': 2.2,           # High - daily restocking
            'Beverages': 1.8,       # High-medium
            'Bakery': 3.0,          # Very high (fresh bread)
            'Pantry': 1.2,          # Medium
            'Frozen Foods': 1.3,    # Medium
            'Health & Beauty': 1.0  # Lower
        }
        
        turnover_factor = base_turnover.get(row['category'], 1.0)

        # Base potential sales calculation
        if stock_quantity <= 0:
            base_potential = min_stock * 0.8 if min_stock > 0 else 15
        else:
            base_potential = stock_quantity

        # Shelf life criticality
        if row['shelf_life_days'] <= 2:  # Ultra fresh
            turnover_factor *= 3.0
        elif row['shelf_life_days'] <= 3:
            turnover_factor *= 2.5
        elif row['shelf_life_days'] <= 7:
            turnover_factor *= 2.0

        # Demand intensity
        demand_boost = {
            'Very High': 1.8,
            'High': 1.5,
            'Normal': 1.0,
            'Low': 0.6
        }
        turnover_factor *= demand_boost.get(str(row['sales_demand']), 1.0)

        # Seasonal and calendar effects
        if row['in_season']:
            turnover_factor *= 1.5
            
        if row['is_holiday']:
            turnover_factor *= 1.6
        elif row['is_weekend']:
            turnover_factor *= 1.3

        # Weather adjustment (milder impact for high-turnover goods)
        weather_multiplier = {
            'Catastrophic': 0.1,
            'Extreme': 0.6,
            'Severe': 0.8,
            'Moderate': 0.95,
            'Normal': 1.0
        }
        turnover_factor *= weather_multiplier.get(row['weather_severity'], 1.0)

        # Calculate final sales potential
        sales_potential = base_potential * turnover_factor
        
        # Add noise and ensure realistic values
        noise = sales_potential * 0.25
        simulated_sales = np.random.normal(sales_potential, noise)
        simulated_sales = max(1, simulated_sales)  # Minimum 1 unit
        
        return int(round(simulated_sales))

    return df.apply(calculate_sales_per_row, axis=1)




def simulate_stock_quantity(row: dict):
    """
    Simulates the stock quantity based on demand and seasonality factors.
    Incorporates randomness and scenarios of stock shortage or surplus.

    Parameters:
        row (dict): A dictionary containing product-related data, including:
            - 'in_season' (bool): Indicates if the product is in season.
            - 'max_stock' (float): Maximum stock level.
            - 'min_stock' (float): Minimum stock level.
            - 'weather_severity' (str): Weather impact level ('High', 'Low', etc.).
            - 'adjusted_demand' (float): Demand adjusted for current conditions.

    Returns:
        int: Simulated stock quantity, ensuring non-negative values.

    Example:
        >>> import numpy as np
        >>> row = {
        ...     'in_season': True,
        ...     'max_stock': 100,
        ...     'min_stock': 30,
        ...     'weather_severity': 'Low',
        ...     'adjusted_demand': 60
        ... }
        >>> simulate_stock_quantity(row)
        45  # (actual output may vary due to randomness)
    """
    # Set base stock depending on seasonality
    if row['in_season']:
        base_stock = row['max_stock'] * 0.8
    else:
        base_stock = row['min_stock'] * 1.5

    # Adjust base stock according to weather severity
    if row['weather_severity'] == 'High':
        base_stock *= 0.7  # Severe weather may hinder restocking
    elif row['weather_severity'] == 'Low':
        base_stock *= 1.1  # Favorable weather may allow higher stock

    # Calculate initial stock and add random noise
    simulated_stock = base_stock - row['adjusted_demand'] + np.random.normal(0, 5)

    # Introduce intentional scenarios of stock shortage or surplus
    rand_val = np.random.rand()
    if rand_val < 0.1:  # 10% chance of stock shortage
        return np.random.randint(0, int(row['min_stock'] * 0.8))
    elif rand_val > 0.95:  # 5% chance of stock surplus
        return np.random.randint(int(row['max_stock'] * 1.2), int(row['max_stock'] * 1.5))
    else:
        # Ensure the final value is non-negative
        return max(0, int(simulated_stock))



def classify_grocery_demand(dates: pd.Series, country: str = 'BR') -> pd.Series:
    """
    Classifies grocery demand levels based on date characteristics and national holidays.

    Parameters:
    ----------
    dates : pd.Series
        A pandas Series of datetime objects representing transaction or delivery dates.
    country : str, optional
        A two-letter country code (ISO 3166-1 alpha-2) used to determine holidays.
        Defaults to 'BR' (Brazil).

    Returns:
    -------
    pd.Series
        A Series of strings indicating demand classification for each date:
        - 'High (Holiday)' if the date is a national holiday
        - 'High (Beginning of Month)' if the day is between the 1st and 5th
        - 'High (Weekend)' if the date falls on Saturday or Sunday
        - 'Normal' otherwise

    Example:
    -------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> dates = pd.Series([datetime(2025, 1, 1), datetime(2025, 1, 3), datetime(2025, 1, 4), datetime(2025, 1, 6)])
    >>> classify_grocery_demand(dates, country='BR')
    0         High (Holiday)
    1    High (Beginning of Month)
    2         High (Weekend)
    3                Normal
    dtype: object
    """
    country_holidays = holidays.country_holidays(country)

    def classify_single(date):
        if date in country_holidays:
            return 'Very High'
        elif 1 <= date.day <= 5:
            return 'High'
        elif date.weekday() >= 5:  # Saturday or Sunday
            return 'High'
        else:
            return 'Normal'
    
    return dates.apply(classify_single)





def day_classification(dates: pd.Series, country: str = 'BR') -> pd.Series:
    """
    Classifies each date in a pandas Series as 'Holiday', 'Saturday', 'Sunday', or 'Weekdays' 
    based on the specified country's holiday calendar.

    Parameters:
    ----------
    dates : pd.Series
        A pandas Series of datetime objects to classify.
    country : str, optional
        A two-letter country code (ISO 3166-1 alpha-2) used to determine holidays. 
        Defaults to 'BR' (Brazil).

    Returns:
    -------
    pd.Series
        A Series of strings indicating the classification of each date.

    Example:
    -------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> dates = pd.Series([datetime(2025, 1, 1), datetime(2025, 1, 4), datetime(2025, 1, 5), datetime(2025, 1, 6)])
    >>> day_classification(dates, country='BR')
    0     Holiday
    1    Saturday
    2      Sunday
    3    Weekdays
    dtype: object
    """
    country_holidays = holidays.country_holidays(country)

    def classify_single(date):
        if date in country_holidays:
            return 'Holiday'
        elif date.dayofweek == 5:
            return 'Saturday'
        elif date.dayofweek == 6:
            return 'Sunday'
        else:
            return 'Weekdays'
        
    return dates.apply(classify_single)




def create_stock_distribution_vectorized(stock_min, stock_max, seed: int=None, 
                                         prob_stock: list=[0.12, 0.28, 0.60],
                                         prob_extreme: list=[0.68, 0.27, 0.05]):
    """
    Generates a vector of stock quantities based on probabilistic conditions: 'out of stock', 'overstocked', or 'normal',
    applied element-wise to input vectors.

    Parameters:
    ----------
    stock_min : pandas.Series
        Series of minimum stock quantities for each item.
    stock_max : pandas.Series
        Series of maximum stock quantities for each item.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    prob_stock : list of float, optional
        Probabilities for each stock condition: ['out', 'over', 'normal'] respectively.
        Default is [0.12, 0.28, 0.60].
    prob_extreme : list of float, optional
        Probabilities for selecting intervals within 'out' and 'over' conditions.
        Default is [0.68, 0.27, 0.05].

    Returns:
    -------
    numpy.ndarray
        Array of generated stock quantities for each item.

    Stock Conditions:
    -----------------
    - 'normal': Random integer between stock_min[i] and stock_max[i].
    - 'over': Value above stock_max[i] using a multiplier from a probabilistic interval.
    - 'out': Value below stock_min[i] using a divider from a probabilistic interval.

    Example:
    --------
    >>> import pandas as pd
    >>> stock_min = pd.Series([50, 30, 20])
    >>> stock_max = pd.Series([100, 80, 60])
    >>> create_stock_distribution_vectorized(stock_min, stock_max, seed=42)
    array([  0,  94,  44])  # (actual values may vary depending on condition and seed)
    """

    # Initialize random number generator with optional seed
    rng = np.random.default_rng(seed=seed)
    n = len(stock_min)

    # Define possible stock conditions
    stock_condition = ['out', 'over', 'normal']

    # Randomly assign a condition to each item
    conditions = rng.choice(stock_condition, size=n, p=prob_stock)

    # Initialize result array
    results = np.zeros(n, dtype=int)

    # Process each item based on its assigned condition
    for i, condition in enumerate(conditions):
        if condition == 'normal':
            # Generate stock within normal range
            results[i] = rng.integers(stock_min.iloc[i], stock_max.iloc[i] + 1)

        elif condition == 'over':
            # Define intervals for overstock multipliers
            multipliers_intervals = [(1.05, 1.15), (1.16, 1.30), (1.31, 1.70)]
            # Select an interval based on extreme probabilities
            chosen_interval = rng.choice(multipliers_intervals, p=prob_extreme)
            # Generate multiplier and apply to stock_max
            multiplier = rng.uniform(chosen_interval[0], chosen_interval[1])
            results[i] = int(np.ceil(stock_max.iloc[i] * multiplier))

        elif condition == 'out':
            # Define intervals for out-of-stock dividers
            dividers_intervals = [(0.05, 0.15), (0.16, 0.30), (0.31, 0.70)]
            # Select an interval based on extreme probabilities
            chosen_interval = rng.choice(dividers_intervals, p=prob_extreme)
            # Generate divider and apply to stock_min
            divider = rng.uniform(chosen_interval[0], chosen_interval[1])
            results[i] = int(np.floor(stock_min.iloc[i] * divider))

    return results





def create_suppliers(storage_df:pd.DateOffset, cost_price_df:pd.DataFrame, products_df:pd.DataFrame, suppliers_df:pd.DataFrame):
    """
        Generates a supplier-product relationship table with realistic logistics and commercial attributes.

        This function links suppliers to products based on category, assigning one primary supplier and
        multiple secondary suppliers per product. It uses lead time and shelf life intervals to simulate
        delivery dynamics and applies randomized commercial parameters such as price variation, reliability,
        payment terms, and quality ratings.

        Parameters:
            storage_df (pd.DataFrame): Contains category-level lead time and shelf life intervals.
            cost_price_df (pd.DataFrame): Contains product names and their base supply prices.
            products_df (pd.DataFrame): Contains product metadata including product ID and category.
            suppliers_df (pd.DataFrame): Contains supplier metadata including supplier ID and category.

        Returns:
            pd.DataFrame: A DataFrame with supplier-product relationships, including:
                - supplier_id
                - product_id
                - supply_price
                - is_primary_supplier
                - lead_time_days
                - min_order_quantity
                - reliability_score
                - payment_terms_days
                - quality_rating
    """
    # Function to convert a string interval like '[min, max]' into a tuple (min, max)
    def parse_interval(interval_str):
        """Converts string '[min, max]' to tuple (min, max)"""
        try:
            if isinstance(interval_str, str):
                return ast.literal_eval(interval_str)  # Safely evaluate string to tuple
            return interval_str  # If already a tuple, return as-is
        except:
            return (1, 30)  # Default fallback range

    # Apply interval parsing to lead time and shelf life columns
    storage_df['lead_time_range'] = storage_df['lead_time_days'].apply(parse_interval)
    storage_df['shelf_life_range'] = storage_df['shelf_life_days'].apply(parse_interval)

    # Create dictionaries mapping each category to its lead time and shelf life ranges
    category_lead_times = {}
    category_shelf_lives = {}
    for _, row in storage_df.iterrows():
        category_lead_times[row['category']] = row['lead_time_range']
        category_shelf_lives[row['category']] = row['shelf_life_range']

    # Initialize list to store supplier-product relationship records
    supplier_product_relationships = []

    # Iterate through each product in the pricing dataset
    for _, price_row in cost_price_df.iterrows():
        product_name = price_row['product']
        main_supply_price = price_row['price']
        
        # Get product details from products_df
        product_info = products_df[products_df['product'] == product_name]
        if len(product_info) == 0:
            continue  # Skip if product not found
        
        product_id = product_info['product_id'].values[0]
        product_category = product_info['category'].values[0]
        
        # Get lead time range for the product's category
        lead_time_range = category_lead_times.get(product_category, (1, 10))
        
        # Filter suppliers that match the product category
        category_suppliers = suppliers_df[suppliers_df['category'] == product_category]

        # Initialize random number generator with fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)
        
        if len(category_suppliers) > 0:
            # Sort suppliers to consistently select the first as the main supplier
            category_suppliers = category_suppliers.sort_values('supplier_id')
            
            # Assign main supplier
            main_supplier = category_suppliers.iloc[0]
            main_supplier_id = main_supplier['supplier_id']
            
            # Generate lead time for main supplier within a narrow range
            main_lead_time = rng.integers(lead_time_range[0], lead_time_range[0] + 3)
            
            # Add main supplier relationship entry
            supplier_product_relationships.append({
                'supplier_id': main_supplier_id,
                'product_id': product_id,
                'supply_price': main_supply_price,
                'is_primary_supplier': True,
                'lead_time_days': main_lead_time,
                'min_order_quantity': rng.choice([3, 5, 10]),
                'reliability_score': round(rng.uniform(0.85, 0.95), 2),
                'payment_terms_days': 30,
                'quality_rating': rng.choice([4, 5], p=[0.3, 0.7])
            })
            
            # Process secondary suppliers
            secondary_suppliers = category_suppliers.iloc[1:]
            
            for _, supplier in secondary_suppliers.iterrows():
                supplier_id = supplier['supplier_id']
                supplier_name = supplier['supplier']
                
                # Define valid lead time range for secondary suppliers
                valid_range = range(max(1, lead_time_range[0] + 1), lead_time_range[1])
                
                # Adjust price and lead time based on supplier type
                if 'Distributor' in supplier_name:
                    variation_factor = rng.uniform(0.95, 1.15)
                    lead_time = rng.choice(valid_range)
                else:
                    variation_factor = rng.uniform(0.9, 1.2)
                    lead_time = rng.integers(lead_time_range[0], lead_time_range[1])
                
                # Calculate secondary supplier price with bounds
                secondary_price = round(main_supply_price * variation_factor, 2)
                secondary_price = max(secondary_price, main_supply_price * 0.5)
                secondary_price = min(secondary_price, main_supply_price * 2.0)
                
                # Add secondary supplier relationship entry
                supplier_product_relationships.append({
                    'supplier_id': supplier_id,
                    'product_id': product_id,
                    'supply_price': secondary_price,
                    'is_primary_supplier': False,
                    'lead_time_days': lead_time,
                    'min_order_quantity': rng.choice([5, 10, 25, 50]),
                    'reliability_score': round(rng.uniform(0.7, 0.9), 2),
                    'payment_terms_days': rng.choice([15, 30, 45, 60]),
                    'quality_rating': rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
                })

    # Convert the list of relationships into a DataFrame
    supplier_products_df = pd.DataFrame(supplier_product_relationships)

    # Return the final DataFrame
    return supplier_products_df




def random_shelf_life(range_list):
    """
        Generate a random integer shelf life value within a specified range.

        Parameters:
            range_list (list or tuple): A two-element list or tuple containing the minimum and maximum 
                                        shelf life values (e.g., [10, 30] or (10, 30)).

        Returns:
            int: A randomly selected integer between the lower bound (inclusive) and upper bound (exclusive).

        Example:
            random_shelf_life([10, 30]) → 17
    """
    
    # Create a random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    
    return rng.integers(range_list[0], range_list[1])





def create_supplier_cat(categories: list, categories_prob: list, supplier_pool: list) -> dict:
    """
    Assigns a random category to each supplier based on a given probability distribution.

    Parameters:
    - categories (list): A list of supply chain categories (e.g., 'Produce', 'Dairy and Cold Cuts').
    - categories_prob (list): A list of probabilities corresponding to each category. Must sum to 1.
    - supplier_pool (list): A list of supplier names.

    Returns:
    - dict: A dictionary mapping each supplier to a randomly assigned category.
    """

    # Remove duplicates, if any
    supplier_pool = list(set(supplier_pool))

    # Create dictionary with randomly assigned categories based on probabilities
    supplier_dict = {
        supplier: np.random.choice(categories, p=categories_prob)
        for supplier in supplier_pool
    }

    return supplier_dict




def create_IDs(length: int, suffix: str):
    """
    Generates a list of unique identifiers in the format 'number|suffix'.

    The IDs are created by adding a base value (1e6) to a random sample of unique
    integers in the range [0, 1e6). The suffix is appended to each number using the '|' separator.

    Parameters:
    ----------
    length : int
        Number of unique IDs to generate.
    suffix : str
        Suffix to be appended to each ID.

    Returns:
    -------
    np.ndarray
        Array of strings containing the unique IDs in the format 'number|suffix'.
    """

    # Base value to ensure large and consistent numbers
    base_id = int(1e6)

    # Generate 'length' unique random integers in the range [0, 1e6)
    timestamps = np.random.choice(int(1e6), size=length, replace=False)

    # Add the base value to the generated numbers to create numeric IDs
    number = base_id + timestamps

    # Format each number as a string with the suffix, separated by '|'
    return np.array([f'{num}|{suffix}' for num in number])
