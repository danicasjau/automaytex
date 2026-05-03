# automaytex\cModels.py
# models control

class cModels():
    def __init__(self):
        self.diffusion_model = None

    def get_model_data(self, configuration=None):
        if configuration == None:
            from config import configuration
            configuration = configuration()
        
    
    def load_all(self, configuration):
        self.models2load = self.config.base_model
        if self.diffusion_model == None:
            print("Loading diffusion model")
        pass
    
    def unload_all(self):
        print("Unloading models")
        pass
    
    def install_models(self):
        print("Installing models")
        pass