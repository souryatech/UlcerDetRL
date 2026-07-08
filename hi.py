import json

with open("hyper_kvasir_mock/bounding_boxes.json") as f:
    data = json.load(f)

migrated = {
    fname: (entry if isinstance(entry, list) else [entry])
    for fname, entry in data.items()
}

with open("hyper_kvasir_mock/bounding_boxes.json", "w") as f:
    json.dump(migrated, f, indent=2)