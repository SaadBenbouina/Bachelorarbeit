import json

# Load the JSON file
input_json = 'combined_panoptic_annotations.json'
output_json = 'combined_panoptic_annotations_fixed.json'

with open(input_json, 'r') as f:
    coco_data = json.load(f)

# New annotations list to store corrected structure
new_annotations = []

# Flatten segments_info into individual annotations
annotation_id = 1
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    for segment in annotation['segments_info']:
        new_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": segment['category_id'],
            "bbox": segment['bbox'],
            "area": segment['area'],
            "iscrowd": segment['iscrowd']
        }
        new_annotations.append(new_annotation)
        annotation_id += 1

# Replace the old annotations with the new flattened annotations
coco_data['annotations'] = new_annotations

# Save the fixed JSON file
with open(output_json, 'w') as f:
    json.dump(coco_data, f)

print(f"Fixed COCO annotations saved to {output_json}")
