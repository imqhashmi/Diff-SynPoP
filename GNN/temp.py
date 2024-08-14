import math

# Original observed counts
observed_counts = {
    'Child-Male-White': 0,
    'Child-Male-Black': 0,
    'Child-Male-Asian': 0,
    'Child-Female-White': 2,
    'Child-Female-Black': 0,
    'Child-Female-Asian': 1,
    'Adult-Male-White': 1,
    'Adult-Male-Black': 2,
    'Adult-Male-Asian': 1,
    'Adult-Female-White': 0,
    'Adult-Female-Black': 0,
    'Adult-Female-Asian': 0,
    'Elder-Male-White': 0,
    'Elder-Male-Black': 0,
    'Elder-Male-Asian': 1,
    'Elder-Female-White': 1,
    'Elder-Female-Black': 1,
    'Elder-Female-Asian': 0
}

# Total persons to scale up to
k = 1000

# Calculate the total count in the original dictionary
total_original_count = sum(observed_counts.values())

# Calculate the scaling factor
scaling_factor = k / total_original_count

# Scale up the counts
scaled_counts = {key: math.floor(value * scaling_factor) for key, value in observed_counts.items()}

# Adjust to ensure the total is exactly k
scaled_total = sum(scaled_counts.values())
adjustment_needed = k - scaled_total

# Make adjustments to ensure the total is exactly k
for key in sorted(scaled_counts, key=scaled_counts.get, reverse=True):
    if adjustment_needed <= 0:
        break
    scaled_counts[key] += 1
    adjustment_needed -= 1

print("Scaled counts:", scaled_counts)
print("Total scaled count:", sum(scaled_counts.values()))