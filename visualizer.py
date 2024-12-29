# visualizer.py
import matplotlib.pyplot as plt
import torch
import streamlit as st

class Visualizer:
   @staticmethod
   def plot_loss(epochs, loss_history):
       plt.figure(figsize=(10, 5))
       plt.plot(range(1, epochs + 1), loss_history, marker='o')
       plt.title('Training Loss Progression')
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.grid(True)
       plt.tight_layout()
       st.pyplot(plt)

   @staticmethod
   def show_predictions(model, device, train_loader, class_mapping):
       model.eval()
       examples = enumerate(train_loader)
       batch_idx, (example_data, example_targets) = next(examples)
       example_data = example_data.to(device, dtype=torch.float32)
       
       with torch.no_grad():
           output = model(example_data)

       fig, axes = plt.subplots(1, 10, figsize=(15, 2))
       for i in range(10):
           ax = axes[i]
           # Reshape from (3,28,28) to (28,28,3) and scale back to 0-1 range
           img = example_data[i].cpu().numpy().transpose(1, 2, 0)
           ax.imshow(img)
           pred = output[i].argmax(dim=0, keepdim=True).item()
           # Get actual class name for prediction
           pred_class = [k for k, v in class_mapping.items() if v == pred][0]
           ax.set_title(f'Pred: {pred_class}')
           ax.axis('off')
       st.pyplot(fig)
