import matplotlib.pyplot as plt

# Mock data for training/validation accuracy
epochs = list(range(1, 21))
train_acc = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92, 0.93,
             0.94, 0.95, 0.955, 0.96, 0.965, 0.97, 0.973, 0.975, 0.978, 0.98]
val_acc = [0.60, 0.68, 0.75, 0.79, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90,
           0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='s')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()

# ✅ Save to image file
plt.savefig("accuracy_graph.png")  # You can change the name or format
plt.show()
