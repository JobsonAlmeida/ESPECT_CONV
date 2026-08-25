from pathlib import Path

import torch

import matplotlib.pyplot as plt

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_40_cnn"
)

MODEL_DIR = INPUT_DIR / "space_frequency_models"/ "sub-01"

checkpoint = torch.load(
    MODEL_DIR / "space_frequency_fold_1.pth",
    weights_only=False
)

train_acc = checkpoint[
    "train_accuracies_epoch"
]

test_acc = checkpoint[
    "test_accuracies_epoch"
]

train_loss = checkpoint[
    "train_losses_epoch"
]

test_loss = checkpoint[
    "test_losses_epoch"
]

epochs = range(
    1,
    len(train_acc) + 1
)


fig, axs = plt.subplots(2, 6, figsize=(18, 6))


axs[0, 0].plot(epochs, train_loss, color='blue')


axs[0, 0].plot(
    epochs,
    test_loss,
    label="Test loss"
)

# axs[0, 1].plot(
#     epochs,
#     train_acc,
#     label="Train accuracy"
# )

# axs[0, 1].plot(
#     epochs,
#     test_acc,
#     label="Test accuracy"
# )


# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid()

plt.tight_layout()

plt.show()