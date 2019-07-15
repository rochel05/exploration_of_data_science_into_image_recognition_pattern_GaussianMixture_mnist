from model import GaussianMixtureClassifierModel
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from load_data_from_csv import load_data_from_csv
from load_data_from_mongoDb import load_data_from_mongoDb
from load_data_from_rawData_Img import imageLoader
from load_data_from_rawData_txt import sentenceLoader
from keras.utils import to_categorical
import numpy as np
from pickle import load, dump
from .model import agregation_of_heterogenous_datas, reduction_of_dimension_with_PCA, reduction_of_dimension_with_LDA

def test():
    #load heterogenous data
    Xtrain1, Xtest1 = imageLoader()
    Ytrain1, Ytest1 = sentenceLoader()
    df_test_x, df_test_y, df_train_x, df_train_y = load_data_from_csv()
    Xtrain2, Xtest2a, Ytrain2, Ytest2a = train_test_split(df_train_x, df_train_y, random_state=0, test_size=0.9)
    Xtrain2b, Xtest2, Ytrain2b, Ytest2 = train_test_split(df_test_x, df_test_y, random_state=0, test_size=0.9)
    df_train_xM, df_train_yM = load_data_from_mongoDb()
    Xtrain3, Xtest3, Ytrain3, Ytest3 = train_test_split(df_train_xM, df_train_yM, random_state=0, test_size=0.9)

    #agregate data with numpy
    Xtrain, Ytrain, Xtest, Ytest = agregation_of_heterogenous_datas(Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, Xtest1, Ytest1, Xtest2, Ytest2, Xtest3, Ytest3)
    #reduce dimension of agregated data with PCA
    Xtrain, Xtest = reduction_of_dimension_with_PCA(Xtrain, Xtest)
    # reduce dimension of agregated data with LDA
    #Xtrain, Xtest = reduction_of_dimension_with_LDA(Xtrain, Xtest, Ytrain)

    #call model
    GMixtureClassifier = GaussianMixtureClassifierModel()
    #fit data into GMixtureClassifier
    GMixtureClassifierTrain = GMixtureClassifier.fit(Xtrain, Ytrain)
    #evaluate
    predicted = GMixtureClassifier.predict(Xtest)
    print('accuracy : {}%'.format(str(accuracy_score(Ytest, predicted)*100)))
    print(classification_report(Ytest, predicted))
    print('score train : ', GMixtureClassifier.score(Xtest, Ytest))
    #print(confusion_matrix(Ytest, predicted))
    with open("model/GMixtureClassifier_pca_csv_raw_test.pkl", "wb") as fichier:
        dump(GMixtureClassifierTrain, fichier)

if __name__=='__main__':
    test()

    #df_test_x = df_test_x.astype('int16')
    #df_test_y = df_test_y.astype('int16')
    #df_train_x = df_train_x.astype('int16')
    #df_train_y = df_train_y.astype('int16')