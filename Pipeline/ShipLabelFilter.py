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
        if 'Great Lakes Tugs, Barges and Work Boats' in label:
            label = None
        if 'Supply Ships/Tug Supplies/AHTS' in label:
            label = None
        if 'mystery' in label:
            label = None
        if label is None:
            return "label"

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
        if 'cargo_ship' in label:
            label = 'cargo_ship'
        if 'container' in label:
            label = 'cargo_ship'
        if 'tank' in label:
            label = 'cargo_ship'
        if 'bulk' in label:
            label = 'cargo_ship'
        if 'reefer' in label:
            label = 'cargo_ship'
        if 'tanker' in label:
            label = 'cargo_ship'
        if 'vehicle_carrier' in label:
            label = 'cargo_ship'
        if 'barge' in label:
            label = 'cargo_ship'
        if 'ro' in label:
            label = 'cargo_ship'
        if 'general_cargo' in label:
            label = 'cargo_ship'
        if 'landing_craft' in label:
            label = 'cargo_ship'
        if 'cement' in label:
            label = 'cargo_ship'
        if 'aggregates_carrier' in label:
            label = 'cargo_ship'
        if 'vehicle' in label:
            label = 'cargo_ship'
        if 'ore' in label:
            label = 'cargo_ship'
        if 'dry_cargo' in label:
            label = 'cargo_ship'

        if 'drill' in label:
            label = 'work_boat'
        if 'tug' in label:
            label = 'work_boat'
        if 'work_boat' in label:
            label = 'work_boat'

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
        if 'attack' in label:
            label = 'military'
        if 'landing_ship' in label:
            label = 'military'

        if 'lifeboat' in label:
            label = 'rescue_boat'
        if 'rescue' in label:
            label = 'rescue_boat'

        if 'passenger' in label:
            label = 'ferry'
        if 'ferr' in label:
            label = 'ferry'
        if 'tour' in label:
            label = 'ferry'
        if 'high_speed' in label:
            label = 'ferry'
        if 'steam' in label:
            label = 'ferry'
        if 'cruise' in label:
            label = 'ferry'

        if 'coast_guard' in label:
            label = 'order'
        if 'police' in label:
            label = 'order'
        if 'customs' in label:
            label = 'order'
        if 'patrol' in label:
            label = 'order'

        if 'whale' in label:
            label = 'fishing_ship'
        if 'fish' in label:
            label = 'fishing_ship'

        if 'submarine' in label:
            label = 'submarine'

        if 'sailing' in label:
            label = 'sailing_ship'

        return label
