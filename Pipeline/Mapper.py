ship_mapping = [
    "cargo_ship", "ferry", "fishing_ship", "military",
    "order", "rescue_boat", "sailing_ship",
    "submarine", "work_boat"
]


# Define the mapping function
def map_number_to_ship(number):
    if 0 <= number < len(ship_mapping):
        return ship_mapping[number]
    else:
        return "other"
