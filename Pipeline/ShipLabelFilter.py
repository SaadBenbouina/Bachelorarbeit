class ShipLabelFilter:
    CATEGORY_MAPPING = {
        'cargoship': 'cargo_ship',
        'livestock': 'cargo_ship',
        'combined': 'cargo_ship',
        'crew_vessel': 'work_boat',
        'wood_chip': 'cargo_ship',
        'service_craft': 'military',
        'sd_14': 'cargo_ship',
        'drill': 'work_boat',
        'casult': 'maintenance',
        'buoy': 'buoy',
        'battle': 'military',
        'auxiliar': 'military',
        'frigate': 'military',
        'destroyer': 'military',
        'carrier': 'military',
        'lifeboat': 'rescue_boat',
        'wrecks': 'maintenance',
        'cargo_ship': 'cargo_ship',
        'container': 'cargo_ship',
        'tank': 'tanker',
        'bulk': 'cargo_ship',
        'fish': 'fishing_ship',
        'cruise': 'ferry',
        'reefer': 'cargo_ship',
        'tanker': 'tanker',
        'sailing': 'sailing_ship',
        'motor': 'motor_ship',
        'tug': 'tug',
        'vehicle_carrier': 'cargo_ship',
        'work': 'work_boat',
        'passenger': 'ferry',
        'ferr': 'ferry',
        'barge': 'cargo_ship',
        'rescue': 'rescue_boat',
        'attack': 'military',
        'ro': 'cargo_ship',
        'scrap': 'maintenance',
        'repair': 'maintenance',
        'construction': 'maintenance',
        'tour': 'ferry',
        'cement': 'cargo_ship',
        'submarine': 'submarine',
        'general_cargo': 'cargo_ship',
        'landing_craft': 'cargo_ship',
        'landing_ship': 'military',
        'high_speed': 'ferry',
        'reclassified': 'validation',
        'aggregates_carrier': 'cargo_ship',
        'steam': 'ferry',
        'storm': 'fishing_ship',
        'research': 'special_ship',
        'special': 'special_ship',
        'mystery': 'sailing_ship',
        'training': 'special_ship',
        'support': 'special_ship',
        'heavy': 'special_ship',
        'cable': 'special_ship',
        'coast_guard': 'order',
        'police': 'order',
        'customs': 'order',
        'patrol': 'order',
        'vehicle': 'cargo_ship',
        'ice': 'special_ship',
        'ore': 'cargo_ship',
        'test': 'special_ship',
        'whale': 'fishing_ship',
        'museum': 'validation',
        'shipping': 'validation',
        'dry_cargo': 'cargo_ship'
    }

    EXCLUSION_TERMS = ['overview', 'wheelhouse', 'deck', 'museum', 'arma', 'interior', 'crests']

    @staticmethod
    def filter_label(label):
        """
        Filters and standardizes the given label.

        Args:
            label (str): The label to be filtered.

        Returns:
            str: The filtered and standardized label.
        """
        label = label.lower().replace(" ", "")

        for term in ShipLabelFilter.EXCLUSION_TERMS:
            if term in label:
                return None

        for key, value in ShipLabelFilter.CATEGORY_MAPPING.items():
            if key in label:
                return value

        return None  # Rückgabe des ursprünglichen Labels, falls keine Übereinstimmung gefunden wird
