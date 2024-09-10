import json

# Path to your existing COCO annotation file
input_annotation_file = "/Users/saadbenboujina/Downloads/semantic segmentation-2/train/_annotations.coco.json"
output_annotation_file = '/Users/saadbenboujina/Downloads/semantic segmentation-2/new/output_annotations.coco.json'

# Load the existing annotation file
with open(input_annotation_file, 'r') as f:
    coco_data = json.load(f)

# Categories that should be treated as "stuff" (semantic) instead of "things" (instance)
stuff_categories = ['sky', 'water']

# Find category ids for the stuff categories
stuff_category_ids = [category['id'] for category in coco_data['categories'] if category['name'] in stuff_categories]

# Mark 'sky' and 'water' as semantic (stuff) in the categories
for category in coco_data['categories']:
    if category['name'] in stuff_categories:
        category['isthing'] = 0  # Mark as 'stuff' (semantic)
    else:
        category['isthing'] = 1  # Mark others as 'things' (instance)

# Function to merge segmentation for semantic classes (stuff)
def merge_segments_for_semantic(annotations, stuff_category_ids):
    semantic_annotations = []
    for image_id, image_annotations in annotations.items():
        merged_segments = {}
        for annotation in image_annotations:
            if annotation['category_id'] in stuff_category_ids:
                # Merge all segments for the same semantic category
                if annotation['category_id'] not in merged_segments:
                    merged_segments[annotation['category_id']] = annotation
                else:
                    # Add new segmentation points
                    merged_segments[annotation['category_id']]['segmentation'] += annotation['segmentation']
                    # Update the area
                    merged_segments[annotation['category_id']]['area'] += annotation['area']

        # Add merged segments to semantic_annotations
        semantic_annotations += list(merged_segments.values())
    return semantic_annotations

# Group annotations by image_id
annotations_by_image = {}
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(annotation)

# Merge the segments for semantic classes
semantic_annotations = merge_segments_for_semantic(annotations_by_image, stuff_category_ids)

# Filter out instance-level annotations for semantic classes (sky and water)
instance_annotations = [annotation for annotation in coco_data['annotations'] if annotation['category_id'] not in stuff_category_ids]

# Combine instance and merged semantic annotations
coco_data['annotations'] = instance_annotations + semantic_annotations

# Save the updated annotation file
with open(output_annotation_file, 'w') as f:
    json.dump(coco_data, f, indent=4)

print("Updated annotations have been saved to:", output_annotation_file)
