from roboflow import Roboflow

rf = Roboflow(api_key="PzvehakfMVctMmPa5Fn1")
project = rf.workspace("thomas-febr7").project("face-d7xbs")
version = project.version(1)
dataset = version.download("yolov11")
                
