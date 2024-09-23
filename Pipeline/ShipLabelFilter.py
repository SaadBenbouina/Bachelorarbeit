
class ShipLabelFilter:
    def __init__(self):
        pass

    @staticmethod
    def filter_label(label):
        """
        Filters and standardizes the given label.

        Args:
            label (str): The label to be filtered.

        Returns:
            str: The filtered and standardized label.
        """
        label = label.lower().replace(" ", "_")
        if 'overview' in label:
            label = None
        if 'wheelhouse' in label:
            label = None
        if 'deck' in label:
            label = None
        if 'museum' in label:
            label = None
        if 'arma' in label:
            label = None
        if 'interior' in label:
            label = None
        if 'crests' in label:
            label = None
        if label is None:
            return label

        if 'livestock' in label:
            label = 'cargo_ship'
        if 'combined' in label:
            label = 'cargo_ship'
        if 'crew_vessel' in label:
            label = 'work_boat'
        if 'wood_chip' in label:
            label = 'cargo_ship'
        if 'service_craft' in label:
            label = 'military'
        if 'sd_14' in label:
            label = 'cargo_ship'
        if 'drill' in label:
            label = 'work_boat'
        if 'casult' in label:
            label = 'maintenance'
        if 'buoy' in label:
            label = 'buoy'
        if 'battle' in label:
            label = 'military'
        if 'auxiliar' in label:
            label = 'military'
        if 'frigate' in label:
            label = 'military'
        if 'destroyer' in label:
            label = 'military'
        if 'carrier' in label:
            label = 'military'
        if 'lifeboat' in label:
            label = 'rescue_boat'
        if 'wrecks' in label:
            label = 'maintenance'
        if 'cargo_ship' in label:
            label = 'cargo_ship'
        if 'container' in label:
            label = 'cargo_ship'
        if 'tank' in label:
            label = 'tanker'
        if 'bulk' in label:
            label = 'cargo_ship'
        if 'fish' in label:
            label = 'fishing_ship'
        if 'cruise' in label:
            label = 'ferry'
        if 'reefer' in label:
            label = 'cargo_ship'
        if 'tanker' in label:
            label = 'tanker'
        if 'sailing' in label:
            label = 'sailing_ship'
        if 'motor' in label:
            label = 'motor_ship'
        if 'tug' in label:
            label = 'tug'
        if 'vehicle_carrier' in label:
            label = 'cargo_ship'
        if 'work' in label:
            label = 'work_boat'
        if 'passenger' in label:
            label = 'ferry'
        if 'ferr' in label:
            label = 'ferry'
        if 'barge' in label:
            label = 'cargo_ship'
        if 'rescue' in label:
            label = 'rescue_boat'
        if 'attack' in label:
            label = 'military'
        if 'ro' in label:
            label = 'cargo_ship'
        if 'scrap' in label:
            label = 'maintenance'
        if 'repair' in label:
            label = 'maintenance'
        if 'construction' in label:
            label = 'maintenance'
        if 'tour' in label:
            label = 'ferry'
        if 'cement' in label:
            label = 'cargo_ship'
        if 'submarine' in label:
            label = 'submarine'
        if 'general_cargo' in label:
            label = 'cargo_ship'
        if 'landing_craft' in label:
            label = 'cargo_ship'
        if 'landing_ship' in label:
            label = 'military'
        if 'high_speed' in label:
            label = 'ferry'
        if 'reclassified' in label:
            label = 'validation'
        if 'aggregates_carrier' in label:
            label = 'cargo_ship'
        if 'steam' in label:
            label = 'ferry'
        if 'storm' in label:
            label = 'fishing_ship'
        if 'research' in label:
            label = 'special_ship'
        if 'special' in label:
            label = 'special_ship'
        if 'mystery' in label:
            label = 'sailing_ship'
        if 'training' in label:
            label = 'special_ship'
        if 'support' in label:
            label = 'special_ship'
        if 'heavy' in label:
            label = 'special_ship'
        if 'cable' in label:
            label = 'special_ship'
        if 'coast_guard' in label:
            label = 'order'
        if 'police' in label:
            label = 'order'
        if 'customs' in label:
            label = 'order'
        if 'patrol' in label:
            label = 'order'
        if 'vehicle' in label:
            label = 'cargo_ship'
        if 'ice' in label:
            label = 'special_ship'
        if 'ore' in label:
            label = 'cargo_ship'
        if 'test' in label:
            label = 'special_ship'
        if 'whale' in label:
            label = 'fishing_ship'
        if 'museum' in label:
            label = 'validation'
        if 'shipping' in label:
            label = 'validation'
        if 'dry_cargo' in label:
            label = 'cargo_ship'
        if 'dredgers' in label:
            label = 'maintenance'
        if 'inland_dry_cargo' in label:
            label = 'cargo_ship'
        if 'pilot_vessel' in label:
            label = 'work_boat'
        if 'corvettes' in label:
            label = 'military'
        return label
