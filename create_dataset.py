from PIL import Image
import numpy as np
import pandas as pd

image_size = 224

csvdata = pd.read_csv("archive/idrid_labels.csv")
print(csvdata)
data_len = len(csvdata)
datadir='archive/Imagenes/Imagenes/'

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(data_len):
    filename = datadir + csvdata["id_code"][i] + ".jpg"
    image = Image.open(filename)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)

    if "test" in filename:
        X_test.append(data)
        y_test.append(csvdata["label"][i])
    else:
        X_train.append(data)
        y_train.append(csvdata["label"][i])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

print(X_train)

xy = (X_train, X_test, y_train, y_test)
np.save("./dataset.npy", xy)