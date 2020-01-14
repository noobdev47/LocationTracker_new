import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import random

def test(Ax, Ay, Az, Mx, My, Mz, Gx, Gy, Gz):
    dataset = pd.read_csv('C:/Users/Sumair Saif/Downloads/Pocket.csv', encoding='mac_roman', low_memory=False)

    #dataset = dataset.loc[dataset['Activity_Name'] == name]

    encoder_f = preprocessing.LabelEncoder()

    encoder_Ax = preprocessing.LabelEncoder()

    encoder_Ay = preprocessing.LabelEncoder()

    encoder_Az = preprocessing.LabelEncoder()

    encoder_Mx = preprocessing.LabelEncoder()

    encoder_My = preprocessing.LabelEncoder()

    encoder_Mz = preprocessing.LabelEncoder()

    encoder_Gx = preprocessing.LabelEncoder()

    encoder_Gy = preprocessing.LabelEncoder()

    encoder_Gz = preprocessing.LabelEncoder()

    ax = encoder_Ax.fit_transform(dataset["Ax"].head(244))

    ay = encoder_Ay.fit_transform(dataset["Ay"].head(244))

    az = encoder_Az.fit_transform(dataset["Az"].head(244))

    mx = encoder_Mx.fit_transform(dataset["Mx"].head(244))

    my = encoder_My.fit_transform(dataset["My"].head(244))

    mz = encoder_Mz.fit_transform(dataset["Mz"].head(244))

    gx = encoder_Gx.fit_transform(dataset["Gx"].head(244))

    gy = encoder_Gy.fit_transform(dataset["Gy"].head(244))

    gz = encoder_Gz.fit_transform(dataset["Gz"].head(244))

    activity = encoder_f.fit_transform(dataset["Activity_Label"].head(244))

    features = zip(ax, ay, az, mx, my, mz, gx, gy, gz)

    features = list(features)

    model = KNeighborsClassifier(k=1)
    model.fit(features, activity)

    predicted = model.predict(features[0])

    return str(encoder_f.inverse_transform(predicted))