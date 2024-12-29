# config.py
class ModelConfig:
    @staticmethod
    def get_default_config():
        return {
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 10
        }