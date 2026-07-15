"""One-off check: does Arnold Schwarzenegger have enough images in the
sklearn LFW dataset to be a workable target identity?

Run once, read the output, then delete/ignore -- not part of the pipeline.
"""
from sklearn.datasets import fetch_lfw_people

# min_faces_per_person=1 pulls in everyone so we can search by name.
data = fetch_lfw_people(min_faces_per_person=1, resize=0.4)

names = data.target_names
counts = {}
for label in data.target:
    name = names[label]
    counts[name] = counts.get(name, 0) + 1

matches = {name: c for name, c in counts.items() if "schwarzenegger" in name.lower()}
print("Schwarzenegger matches:", matches)

top20 = sorted(counts.items(), key=lambda kv: -kv[1])[:20]
print("\nTop 20 most-represented people in LFW:")
for name, c in top20:
    print(f"  {c:4d}  {name}")

print(f"\nTotal people in dataset: {len(counts)}")
print(f"Total images: {sum(counts.values())}")
