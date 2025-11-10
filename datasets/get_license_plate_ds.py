from roboflow import Roboflow

# I am leaving private key in here for help on other devices as well,
# will replace it later with .env once it gets public.

rf = Roboflow(api_key="PzvehakfMVctMmPa5Fn1")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(11)
dataset = version.download("yolov11")
                
