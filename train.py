from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n_ECA_CoordA.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n_ECA_CoordA.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="CUHKOcclusion.yaml", 
#                       epochs=400, 
#                       imgsz=640,
#                       batch=8,
#                       optimizer='Adam',
#                       lr0=1e-3,
#                       project='CUHKOcclusion')

# Train the model
results = model.train(data="CityPerson.yaml", 
                      epochs=200, 
                      imgsz=640,
                      batch=8,
                      optimizer='Adam',
                      lr0=1e-3,
                      project='CityPerson_111')

# results = model.train(data="CUHKOcclusion.yaml", 
#                       epochs=400, 
#                       imgsz=640,
#                       batch=8,
#                       optimizer='Adam',
#                       lr0=1e-3,
#                       project='CUHKOcclusion_baseline')

# results = model.train(data="CityPerson.yaml", 
#                       epochs=200, 
#                       imgsz=640,
#                       batch=8,
#                       optimizer='Adam',
#                       lr0=1e-3,
#                       project='CityPerson_baseline')