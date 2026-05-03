class lModels():
    def __init__(self):
        self.diffusion_model = None
    
    def load_all(self, configuration):
        self.models2load = self.config.base_model
        if self.diffusion_model == None:
            print("Loading diffusion model")
        pass
    
    def unload_all(self):
        print("Unloading models")
        pass
    
    