# This file holds the mean and standard deviation for each task
class Task:
    def __init__(self, name):
        if name == "solar":
            self.mean = [0.507, 0.513, 0.462]
            self.std = [0.172, 0.133, 0.114]
        elif name == "crop-delineation":
            self.mean = [0.238, 0.297, 0.317]
            self.std = [0.187, 0.123, 0.114]
        elif name == "building":
            self.mean = [0.406, 0.428, 0.394]
            self.std = [0.201, 0.183, 0.176]
        elif name == "climate+":
            self.mean = [0.460, 0.440, 0.378]
            self.std = [0.179, 0.139, 0.123]
        elif name == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif name == "geonet_1M_checked": 
            self.mean = [0.309, 0.285, 0.227]
            self.std = [0.241, 0.204, 0.199]
        elif name == "eurosat": 
            self.mean = [0.3440, 0.3801, 0.4076]
            self.std = [0.2021, 0.1362, 0.1149]
        elif name == "ssl4eo":
            self.mean = [0.4838, 0.4808, 0.4780]
            self.std = [0.1964, 0.1733, 0.1484]
        else:
            raise NotImplementedError("Task not implemented")
