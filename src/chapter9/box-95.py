import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Ground truth box (green)
gt_box = patches.Rectangle((2, 3), 4, 4, linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth')
ax.add_patch(gt_box)

# Predicted box (red)
pred_box = patches.Rectangle((3, 4), 4, 4, linewidth=2, edgecolor='r', facecolor='none', label='Prediction')
ax.add_patch(pred_box)

plt.legend()
plt.title("IoU Example: Overlapping Bounding Boxes")
plt.show()
