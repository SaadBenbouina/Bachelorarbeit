import json

# Load the instance and semantic JSON files
with open("export_coco-panoptic_smsabenb_boat2_test.json") as f1, open("export_coco-panoptic_smsabenb_boat2-semantic_vvv.json") as f2:
    instance_data = json.load(f1)
    semantic_data = json.load(f2)

# Create a new structure for the combined panoptic segmentation
combined_data = {
    "info": instance_data["info"],
    "categories": instance_data["categories"],
    "images": instance_data["images"],
    "annotations": []
}

# Iterate over the images and combine the annotations
for instance_ann, semantic_ann in zip(instance_data["annotations"], semantic_data["annotations"]):
    combined_segments_info = instance_ann["segments_info"] + semantic_ann["segments_info"]

    # Add the combined segments to the new annotation
    combined_data["annotations"].append({
        "image_id": instance_ann["image_id"],
        "file_name": instance_ann["file_name"],
        "segments_info": combined_segments_info
    })

# Save the combined data into a new JSON file
with open("combined_panoptic_annotations.json", "w") as f_out:
    json.dump(combined_data, f_out)

print("Combined panoptic annotations have been saved.")
