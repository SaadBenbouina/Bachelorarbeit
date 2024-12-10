
#need to change
ship_mapping = [
     "ferry", "freight_ship", "maintenance", "military",
     "order", "sailing_ship", "small_boat",
    "submarine", "utility_boat"
]

# Define the mapping function
def map_number_to_ship(number):
    if 0 <= number < len(ship_mapping):
        return ship_mapping[number]
    else:
        return "other"


