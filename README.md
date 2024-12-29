# training-image-classifier
An interactive web app that teaches computers to recognize handwritten numbers. Users adjust settings and watch in real-time as the AI learns from thousands of examples, ultimately being able to identify digits. A hands-on way to see machine learning in action.

The code structure is as follows:

    1. model.py: Defines the MNIST CNN architecture. (This will be expanded to include other models)
		
    2. data_loader.py: Handels loading and preprocessing the MNIST dataset
		
    3. config.py: Provides default hyerparameters
		
    4. trainer.py: Contains the training loop logic  
		
    5. visulaizer.py: Handles plotting and visualization
		
    6. app.py: Manges the Streamlit interface and orchestrates the training process.
