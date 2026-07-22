import matplotlib.pyplot as plt
from data_generator import generate_sample, CLASSES

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for row, label in enumerate(CLASSES):
    for col in range(5):
        img = generate_sample(label)
        axes[row][col].imshow(img, cmap="gray")
        axes[row][col].set_title(label if col == 0 else "")
        axes[row][col].axis("off")
plt.tight_layout()
plt.show()
