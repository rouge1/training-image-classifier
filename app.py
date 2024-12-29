# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import MNISTNet
from data_loader import MNISTDataLoader
from config import ModelConfig
from trainer import Trainer
from visualizer import Visualizer
from torchvision import datasets
from torch.utils.data import DataLoader
from inference import ImagePredictor, ModelType
from PIL import Image

class MNISTApp:
    def __init__(self):
        self.setup_seeds()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_sidebar()
        self.model_type = ModelType.MNIST
        
        # Initialize session state if it doesn't exist
        if 'predictor' not in st.session_state:
            st.session_state.predictor = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'class_mapping' not in st.session_state:
            st.session_state.class_mapping = None
        
    @staticmethod
    def setup_seeds():
        torch.manual_seed(42)
        np.random.seed(42)

    def setup_sidebar(self):
        st.sidebar.title("MNIST Training Optimizer")
        config = ModelConfig.get_default_config()
        
        #st.sidebar.subheader("Hyperparameters")
        self.batch_size = st.sidebar.number_input("Batch Size", 
            min_value=1, max_value=512, value=config["batch_size"])
        self.learning_rate = st.sidebar.number_input("Learning Rate", 
            min_value=0.0001, max_value=0.01, value=config["learning_rate"], 
            format="%f")
        self.weight_decay = st.sidebar.number_input("Weight Decay", 
            min_value=0.0, max_value=0.01, value=config["weight_decay"], 
            format="%f")
        self.epochs = st.sidebar.number_input("Training Epochs", 
            min_value=1, max_value=50, value=config["epochs"])
        
        # Add Train Model button here
        self.train_button = st.sidebar.button("Train Model")
        
        # Add spacing
        st.sidebar.markdown("---")  # Adds a horizontal line for separation
        
        # Add inference interface below
        st.sidebar.subheader("Test Your Own Image")
        self.uploaded_file = st.sidebar.file_uploader(
            "Drop an image here...", 
            type=['png', 'jpg', 'jpeg']
        )

    def setup_inference(self, model, class_mapping):
        """Setup the predictor after model training"""
        predictor = ImagePredictor(
            model=model,
            device=self.device,
            class_mapping=class_mapping,
            model_type=self.model_type
        )
        # Store in session state
        st.session_state.predictor = predictor
        st.session_state.model = model
        st.session_state.class_mapping = class_mapping

    def run(self):
        if self.train_button:  # Use the button state from sidebar
            model = MNISTNet().to(self.device)
            model = model.to(self.device, dtype=torch.float32)

            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            criterion = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')
            
            data_loader = MNISTDataLoader(batch_size=self.batch_size)
            class_mapping = data_loader.get_class_mapping()
            train_loader = data_loader.get_train_loader()

            # Create a validation loader
            val_dataset = datasets.MNIST('../data', train=False, download=True, transform=data_loader.transform)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize a Streamlit empty container for dynamic updates
            epoch_status_container = st.status(label=f"Training {self.model_type.value} on '{self.device}'")
            batch_status_bar = st.progress(0, text="")
            
            trainer = Trainer(model, self.device, optimizer, criterion, scaler, batch_status_bar, self.model_type, self.epochs)
            
            for epoch in range(1, self.epochs + 1):
                epoch_loss = trainer.train_epoch(train_loader)
                epoch_status_container.update(label=f"{self.model_type.value} Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # Clear and reassign the batch_status_container
            epoch_status_container.update(state="complete")
            batch_status_bar.empty()
            
            #Show plot of training and show some sample predictions
            Visualizer.plot_loss(self.epochs, trainer.loss_history)            
            Visualizer.show_predictions(model, self.device, train_loader, class_mapping)
 
            #save the model we just created
            torch.save(model.state_dict(), f"{self.model_type.value.lower()}.pth")
            st.success(f"Model '{self.model_type.value.lower()}.pth' saved successfully")
           
            # After training is complete, setup the predictor
            self.setup_inference(model, class_mapping)
            self.model = model
            self.class_mapping = class_mapping

        # Handle inference using session state
        if self.uploaded_file is not None:
            if st.session_state.predictor is None:
                st.sidebar.warning("Please train the model first!")
            else:               
                prediction, error = st.session_state.predictor.predict(self.uploaded_file)
                
                if error:
                    st.error(error)
                else:
                    st.success(
                        f"Most closely matches: {prediction['class_name']} "
                        f"with {prediction['confidence']:.2f}% confidence"
                    )
 
                # image = Image.open(self.uploaded_file)
                # st.image(image,
                #         caption='Uploaded Image', 
                #         use_container_width=True)
 
                # Create a container with fixed height
                with st.container():
                    # Calculate a reasonable height (e.g., 300px)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        image = Image.open(self.uploaded_file)
                        st.image(image,
                                caption='Uploaded Image', 
                                use_container_width=True,
                                width=300)  # This controls the actual size
                    
if __name__ == "__main__":
    app = MNISTApp()
    app.run()