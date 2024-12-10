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
            label = "unkown"
        if 'wheelhouse' in label:
            label = "unkown"
        if 'deck' in label:
            label = "unkown"
        if 'museum' in label:
            label = "unkown"
        if 'arma' in label:
            label = "unkown"
        if 'interior' in label:
            label = "unkown"
        if 'crests' in label:
            label = "unkown"
        if label is None:
            return "unkown"

        if 'livestock' in label:
            label = 'freight_ship'
        if 'combined' in label:
            label = 'freight_ship'
        if 'crew_vessel' in label:
            label = 'utility_ship'
        if 'wood_chip' in label:
            label = 'freight_ship'
        if 'service_craft' in label:
            label = 'military'
        if 'sd_14' in label:
            label = 'freight_ship'
        if 'drill' in label:
            label = 'utility_ship'
        if 'casult' in label:
            label = 'maintenance'
        if 'maintenance' in label:
            label = 'maitenance'
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
            label = 'small_boat'
        if 'wrecks' in label:
            label = 'maintenance'
        if 'cargo_ship' in label:
            label = 'freight_ship'
        if 'container' in label:
            label = 'freight_ship'
        if 'tank' in label:
            label = 'freight_ship'
        if 'bulk' in label:
            label = 'freight_ship'
        if 'fish' in label:
            label = 'utility_ship'
        if 'cruise' in label:
            label = 'ferry'
        if 'reefer' in label:
            label = 'freight_ship'
        if 'tanker' in label:
            label = 'freight_ship'
        if 'sailing' in label:
            label = 'sailing_ship'
        if 'motor' in label:
            label = 'unkown'
        if 'tug' in label:
            label = 'small_boat'
        if 'vehicle_carrier' in label:
            label = 'freight_ship'
        if 'work' in label:
            label = 'utility_ship'
        if 'passenger' in label:
            label = 'ferry'
        if 'ferr' in label:
            label = 'ferry'
        if 'barge' in label:
            label = 'freight_ship'
        if 'rescue' in label:
            label = 'small_boat'
        if 'attack' in label:
            label = 'military'
        if 'ro' in label:
            label = 'freight_ship'
        if 'scrap' in label:
            label = 'maintenance'
        if 'repair' in label:
            label = 'maintenance'
        if 'construction' in label:
            label = 'maintenance'
        if 'tour' in label:
            label = 'ferry'
        if 'cement' in label:
            label = 'freight_ship'
        if 'submarine' in label:
            label = 'submarine'
        if 'general_cargo' in label:
            label = 'freight_ship'
        if 'landing_craft' in label:
            label = 'freight_ship'
        if 'landing_ship' in label:
            label = 'military'
        if 'high_speed' in label:
            label = 'ferry'
        if 'reclassified' in label:
            label = 'unkown'
        if 'aggregates_carrier' in label:
            label = 'freight_ship'
        if 'steam' in label:
            label = 'ferry'
        if 'storm' in label:
            label = 'utility_ship'
        if 'coast_guard' in label:
            label = 'order'
        if 'police' in label:
            label = 'order'
        if 'customs' in label:
            label = 'order'
        if 'patrol' in label:
            label = 'order'
        if 'vehicle' in label:
            label = 'freight_ship'
        if 'ore' in label:
            label = 'freight_ship'
        if 'whale' in label:
            label = 'utility_ship'
        if 'museum' in label:
            label = 'unkown'
        if 'shipping' in label:
            label = 'unkown'
        if 'dry_cargo' in label:
            label = 'cargo_ship'
        if 'dredgers' in label:
            label = 'maintenance'
        if 'inland_dry_cargo' in label:
            label = 'freight_ship'
        if 'pilot_vessel' in label:
            label = 'utility_ship'
        if 'corvettes' in label:
            label = 'military'

        return label
