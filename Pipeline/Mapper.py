

ship_mapping = [
    "buoy", "cargo_ship", "ferry", "fishing_ship", "maintenance", "military",
    "motor_ship", "order", "rescue_boat", "sailing_ship", "special_ship",
    "submarine", "tanker", "tug", "validation", "work_boat"
]

# Define the mapping function
def map_number_to_ship(number):
    if 0 <= number < len(ship_mapping):
        return ship_mapping[number]
    else:
        return "other"


