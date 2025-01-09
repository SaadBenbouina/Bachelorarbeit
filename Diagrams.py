import pandas as pd
from matplotlib import pyplot as plt

def plot_training_results(csv_path):
    # Load the CSV data
    data = pd.read_csv(csv_path)

    # Plot train loss in a separate larger diagram
    plt.figure(figsize=(8, 8))
    plt.plot(data['epoch'], data['train_loss'], label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    loss_image_path = 'train_loss_plot.png'
    plt.savefig(loss_image_path)
    plt.show()

    # Plot train accuracy in a separate larger diagram
    plt.figure(figsize=(12, 8))
    plt.plot(data['epoch'], data['train_acc'], label='Train Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    accuracy_image_path = 'train_accuracy_plot.png'
    plt.savefig(accuracy_image_path)
    plt.show()

    print(f"Plots saved as:\n1. {loss_image_path}\n2. {accuracy_image_path}")

# Example usage:
csv_file_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/trial_7_category/results_trial_7.csv"  # Replace this with your actual CSV file path
plot_training_results(csv_file_path)
