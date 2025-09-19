import pandas as pd
import numpy as np
import ast
import holidays



def simulate_weather(date):
    """
    Simulates weather conditions based on the date, considering seasonal patterns in the Southern Hemisphere.
    
    Parameters:
        date (datetime): A datetime object representing the date for which to simulate the weather.
    
    Returns:
        tuple: A tuple containing:
            - temperature (int): Simulated temperature in degrees Celsius.
            - precipitation (int): Simulated precipitation in millimeters.
            - condition (str): A string describing the general weather condition.
    """
    month = date.month

    # Simulate summer (Dec–Mar)
    if month in [12, 1, 2, 3]:
        temperature = np.random.randint(25, 36)
        precipitation = np.random.choice([0, 5, 10, 20], p=[0.5, 0.25, 0.15, 0.1])
        condition = 'Rain and Heat' if precipitation > 0 else 'Sun and Heat'

    # Simulate autumn (Apr–Jun)
    elif month in [4, 5, 6]:
        temperature = np.random.randint(15, 28)
        precipitation = np.random.choice([0, 2, 5], p=[0.7, 0.2, 0.1])
        condition = 'Rainy' if precipitation > 0 else 'Pleasant'

    # Simulate winter (Jul–Sep)
    elif month in [7, 8, 9]:
        temperature = np.random.randint(10, 22)
        precipitation = np.random.choice([0, 1, 3], p=[0.8, 0.15, 0.05])
        condition = 'Cold and Rainy' if precipitation > 0 else 'Cold'

    # Simulate spring (Oct–Nov)
    else:  # month in [10, 11]
        temperature = np.random.randint(18, 30)
        precipitation = np.random.choice([0, 5, 10], p=[0.6, 0.3, 0.1])
        condition = 'Unstable' if precipitation > 0 else 'Pleasant'

    return temperature, precipitation, condition








def classify_grocery_demand(date):
    """
    Classifies a given date according to typical grocery demand patterns.

    Parameters:
    ----------
    date : datetime-like
        The date to be classified.

    Returns:
    -------
    str
        A string indicating the demand level:
        - 'High Demand (Holiday)' for major holidays like Christmas, New Year's, and Easter
        - 'High Demand (Festa Junina)' for seasonal spikes in June
        - 'High Demand (Beginning of Month)' for payment-related demand between the 1st and 5th
        - 'High Demand (Weekend)' for Saturdays and Sundays
        - 'Normal Demand' for all other dates

    Example:
    --------
    >>> import datetime
    >>> classify_grocery_demand(datetime.date(2025, 12, 25))
    'High Demand (Holiday)'
    """

    # 1. Major holidays (sales peaks)
    important_holidays = {
        '2023-04-09', '2024-03-31', '2025-04-20',  # Easter
        '2023-12-24', '2023-12-25', '2023-12-31', '2024-01-01',  # Christmas and New Year
        '2024-12-24', '2024-12-25', '2024-12-31', '2025-01-01',
        '2025-12-24', '2025-12-25', '2025-12-31', '2026-01-01'
    }

    # 2. Seasonal period (e.g., Festa Junina in June)
    if date.month == 6:
        return 'High Demand (Festa Junina)'

    # 3. Payment cycle (spikes at the beginning of the month)
    if 1 <= date.day <= 5:
        return 'High Demand (Beginning of Month)'

    # 4. Weekends (typical high demand)
    if date.weekday() >= 5:  # Saturday or Sunday
        return 'High Demand (Weekend)'

    # 5. Specific commemorative dates
    if date.strftime('%Y-%m-%d') in important_holidays:
        return 'High Demand (Holiday)'

    # Default case
    return 'Normal Demand'



def day_classification(date):
    """
    Classifies a given date as 'Holiday', 'Saturday', 'Sunday', or 'Weekdays' based on Brazilian calendar.

    Parameters:
    ----------
    date : datetime-like
        The date to be classified.

    Returns:
    -------
    str
        A string indicating the type of day:
        - 'Holiday' if the date is a recognized Brazilian public holiday
        - 'Saturday' if the date falls on a Saturday
        - 'Sunday' if the date falls on a Sunday
        - 'Weekdays' for Monday through Friday (excluding holidays)
    
    Example:
    --------
    >>> import datetime
    >>> day_classification(datetime.date(2025, 9, 7))
    'Holiday'  # Brazilian Independence Day
    """

    # Load Brazilian public holidays
    country_holidays = holidays.CountryHoliday('Brazil')

    # Check if the date is a holiday
    if date in country_holidays:
        return 'Holiday'
    # Check if the date is a Saturday
    elif date.dayofweek == 5:
        return 'Saturday'
    # Check if the date is a Sunday
    elif date.dayofweek == 6:
        return 'Sunday'
    # Otherwise, it's a weekday
    else:
        return 'Weekdays'




def create_stock_distribution_vectorized(stock_min, stock_max, seed=None, 
                                         prob_stock=[0.12, 0.28, 0.60],
                                         prob_extreme=[0.68, 0.27, 0.05]):
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
