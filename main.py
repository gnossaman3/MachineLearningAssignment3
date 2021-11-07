import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def get_data():
    df_income = pd.read_csv('Data/income_evaluation.csv', header=0, skipinitialspace=True)
    df_music = pd.read_csv('Data/music_train.csv', header=0, skipinitialspace=True)

    for col in df_income.columns:
        if df_income[col].dtype == type(df_income['age']):
            i = 0
            for group in df_income[col].unique():
                df_income[col] = df_income[col].replace(group, i)
                i += 1

    df_music = df_music.drop(['Artist Name', 'Track Name'], axis=1).dropna(axis=0)

    income_X = df_income.loc[:, df_income.columns != 'income']
    income_Y = df_income['income']
    music_X = df_music.loc[:, df_music.columns != 'Class']
    music_Y = df_music['Class']

    return income_X, income_Y, music_X, music_Y

def ClusteringTuning(income_X, income_Y, music_X, music_Y, verbose=False):
    if verbose:
        print('Beginning Cluster Tuning...')

    income_n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    music_n_clusters = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    if verbose:
        print('Beginning Clustering Tuning For KMeans on Income Dataset...')

    income_train_scores, income_valid_scores = validation_curve(KMeans(), income_X, income_Y, param_name="n_clusters",
                                                                param_range=income_n_clusters, scoring="accuracy")
    income_train_scores = np.mean(income_train_scores, axis=1)
    #income_valid_scores = np.mean(income_valid_scores, axis=1)

    if verbose:
        print('Performing Elbow Method...')

    income_visualizer = KElbowVisualizer(KMeans(), k=income_n_clusters)

    if verbose:
        print('Done. Beginning Clustering Tuning For KMeans on Music Dataset...')

    music_train_scores, music_valid_scores = validation_curve(KMeans(), music_X, music_Y, param_name="n_clusters",
                                                              param_range=music_n_clusters, scoring="accuracy")
    music_train_scores = np.mean(music_train_scores, axis=1)
    #music_valid_scores = np.mean(music_valid_scores, axis=1)

    if verbose:
        print('Performing Elbow Method...')

    music_visualizer = KElbowVisualizer(KMeans(), k=music_n_clusters)

    if verbose:
        print('Done. Creating Graphs For KMeans...')

    income_visualizer.fit(income_X)
    plt.title('Distortion Score Elbow for KMeans Clustering with Income Dataset')
    plt.savefig(fname='KMeansElbowIncome')
    plt.close()

    music_visualizer.fit(music_X)
    plt.title('Distortion Score Elbow for KMeans Clustering with Music Dataset')
    plt.savefig(fname='KMeansElbowMusic')
    plt.close()

    plt.plot(income_n_clusters, income_train_scores)
    plt.title('KMeans Validation Curve for Income Dataset')
    plt.xlabel('n_clusters')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.savefig(fname='KMeansValidCurveIncome')
    plt.close()

    plt.plot(music_n_clusters, music_train_scores)
    plt.title('KMeans Validation Curve for Music Dataset')
    plt.xlabel('n_clusters')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.savefig(fname='KMeansValidCurveMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Clustering Tuning For Expectation Maximization on Income Dataset...')

    income_train_scores, income_valid_scores = validation_curve(GaussianMixture(), income_X, income_Y,
                                                                param_name="n_components",
                                                                param_range=income_n_clusters, scoring="accuracy")
    income_train_scores = np.mean(income_train_scores, axis=1)
    #income_valid_scores = np.mean(income_valid_scores, axis=1)

    if verbose:
        print('Performing Elbow Method...')

    income_visualizer = KElbowVisualizer(KMeans(), k=income_n_clusters)

    if verbose:
        print('Done. Beginning Clustering Tuning For Expectation Maximization on Music Dataset...')

    music_train_scores, music_valid_scores = validation_curve(GaussianMixture(), music_X, music_Y,
                                                              param_name="n_components",
                                                              param_range=music_n_clusters, scoring="accuracy")
    music_train_scores = np.mean(music_train_scores, axis=1)
    #music_valid_scores = np.mean(music_valid_scores, axis=1)

    if verbose:
        print('Performing Elbow Method...')

    music_visualizer = KElbowVisualizer(KMeans(), k=music_n_clusters)

    if verbose:
        print('Done. Creating Graphs For Expectation Maximization...')

    income_visualizer.fit(income_X)
    plt.title('Distortion Score Elbow for EM Clustering with Income Dataset')
    plt.savefig(fname='EMElbowIncome')
    plt.close()

    music_visualizer.fit(music_X)
    plt.title('Distortion Score Elbow for EM Clustering with Music Dataset')
    plt.savefig(fname='EMElbowMusic')
    plt.close()

    plt.plot(income_n_clusters, income_train_scores)
    plt.title('Expectation Maximization Validation Curve for Income Dataset')
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.savefig(fname='EMValidCurveIncome')
    plt.close()

    plt.plot(music_n_clusters, music_train_scores)
    plt.title('Expectation Maximization Validation Curve for Music Dataset')
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.savefig(fname='EMValidCurveMusic')
    plt.close()

    if verbose:
        print('Clustering Tuning Complete.')

def ClusteringAnalysis(income_k, music_k, income_X, income_Y, music_X, music_Y, verbose=False):
    if verbose:
        print('Beginning Clustering Analysis. Starting with KMeans on Income Dataset...')

    train_sizes = np.linspace(.1, 1.0, 10)

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(KMeans(
        n_clusters=income_k), income_X, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('KMeans Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='KMeansLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('KMeans Fit Times Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='KMeansLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning KMeans Cluster Analysis on Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(KMeans(
        n_clusters=music_k), music_X, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('KMeans Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='KMeansLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('KMeans Fit Times Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='KMeansLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning EM Cluster Analysis on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(GaussianMixture(
        n_components=income_k), income_X, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('Expectation Maximization Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='EMLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('Expectation Maximization Fit Times Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='EMLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning EM Cluster Analysis on Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(GaussianMixture(
        n_components=music_k), music_X, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('Expectation Maximization Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='EMLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('Expectation Maximization Fit Times Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='EMLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Clustering Analysis Complete.')

def ReductionAnalysis(income_X, income_Y, music_X, music_Y, verbose=False):
    if verbose:
        print('Beginning Reduction Analysis, Starting with PCA on Income Data...')

    income_loss = []
    for n in range(1,14):
        pca = PCA(n_components=n)
        income_train = pca.fit_transform(income_X)
        income_proj = pca.inverse_transform(income_train)
        income_loss.append(np.sum((income_X-income_proj)**2, axis=1).mean())
    pca_income = income_train

    if verbose:
        print('Done. Creating PCA Income Graphs...')

    x = np.arange(1,14)
    plt.plot(x, income_loss)
    plt.title('Loss Values for Varying Components in PCA for Income Dataset')
    plt.xlabel('n_components')
    plt.ylabel('Loss')
    plt.savefig(fname='PCAAnalysisIncome')
    plt.close()

    pca = PCA(n_components=2)
    pca.fit(income_X)
    plt.bar([1, 2], pca.explained_variance_)
    plt.title('PCA Explained Variance for Income Dataset')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance')
    plt.savefig(fname='PCAExplainedVarianceIncome')
    plt.close()

    if verbose:
        print('Done. Beginning PCA Analysis on Music Data...')

    music_loss = []
    for n in range(1,15):
        pca = PCA(n_components=n)
        music_train = pca.fit_transform(music_X)
        music_proj = pca.inverse_transform(music_train)
        music_loss.append(np.sum((music_X-music_proj)**2, axis=1).mean())
    pca_music=music_train

    if verbose:
        print('Done. Creating PCA Music Graphs...')

    x = np.arange(1, 15)
    plt.plot(x, music_loss)
    plt.title('Loss Values for Varying Components in PCA for Music Dataset')
    plt.xlabel('n_components')
    plt.ylabel('Loss')
    plt.savefig(fname='PCAAnalysisMusic')
    plt.close()

    pca = PCA(n_components=5)
    pca.fit(music_X)
    plt.bar([1, 2, 3, 4, 5], pca.explained_variance_)
    plt.title('PCA Explained Variance for Music Dataset')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance')
    plt.savefig(fname='PCAExplainedVarianceMusic')
    plt.close()

    if verbose:
        print('Done. Beginning ICA Analysis for Income Data...')

    income_kurtosis = []
    for n in range(1, 14):
        ica = FastICA(n_components=n)
        income_train = ica.fit_transform(income_X)
        income_kurtosis.append(kurtosis(income_train).mean())
    ica_income=income_train

    if verbose:
        print('Done. Creating ICA Income Graphs...')

    x = np.arange(1, 14)
    plt.plot(x, income_kurtosis)
    plt.title('Kurtosis for Varying Component Sizes in ICA for Income Dataset')
    plt.xlabel('n_components')
    plt.ylabel('mean kurtosis')
    plt.savefig(fname='ICAAnalysisIncome')
    plt.close()

    if verbose:
        print('Done. Beginning ICA Analysis on Music Data...')

    music_kurtosis = []
    for n in range(1,15):
        ica = FastICA(n_components=n)
        music_train = ica.fit_transform(music_X)
        music_kurtosis.append(kurtosis(music_train).mean())
    ica_music=music_train

    if verbose:
        print('Done. Creating ICA Music Graphs...')

    x = np.arange(1, 15)
    plt.plot(x, music_kurtosis)
    plt.title('Kurtosis for Varying Component Sizes in ICA for Music Dataset')
    plt.xlabel('n_components')
    plt.ylabel('mean kurtosis')
    plt.savefig(fname='ICAAnalysisMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Radomized Projections Analysis on Income Data...')

    income_error = []
    for n in range(1, 14):
        rp = GaussianRandomProjection(n_components=n)
        rp.fit(income_X)
        components = rp.components_
        p_inverse = np.linalg.pinv(components.T)
        reduced = rp.transform(income_X)
        reconstructed = reduced.dot(p_inverse)
        income_error.append(mean_squared_error(income_X, reconstructed))
    rp_income=rp.fit_transform(income_X)

    if verbose:
        print('Done. Creating Graphs...')

    x = np.arange(1, 14)
    plt.plot(x, income_error)
    plt.title('Reconstruction Error for Randomized Projections on Income Dataset')
    plt.xlabel('n_components')
    plt.ylabel('error')
    plt.savefig(fname='RPAnalysisIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Randomized Projections Analysis on Music Data...')

    music_error = []
    for n in range(1,15):
        rp = GaussianRandomProjection(n_components=n)
        rp.fit(music_X)
        components = rp.components_
        p_inverse = np.linalg.pinv(components.T)
        reduced = rp.transform(music_X)
        reconstructed = reduced.dot(p_inverse)
        music_error.append(mean_squared_error(music_X, reconstructed))
    rp_music=rp.fit_transform(music_X)

    if verbose:
        print('Done. Creating Graphs...')

    x = np.arange(1, 15)
    plt.plot(x, music_error)
    plt.title('Reconstruction Error for Randomized Projections on Music Dataset')
    plt.xlabel('n_components')
    plt.ylabel('error')
    plt.savefig(fname='RPAnalysisMusic')
    plt.close()

    if verbose:
        print('Done. Beginning LDA Analysis on Income Data...')

    income_x_train, income_x_test, income_y_train, income_y_test = train_test_split(income_X, income_Y, test_size=0.3)
    music_x_train, music_x_test, music_y_train, music_y_test = train_test_split(music_X, music_Y, test_size=0.3)
    income_scores = []
    for n in range(1, 2):
        clf = LinearDiscriminantAnalysis(n_components=n)
        clf.fit(income_x_train, income_y_train)
        income_scores.append(clf.score(income_x_test, income_y_test))
    lda_income=clf.fit_transform(income_X, income_Y)

    if verbose:
        print('Done. Creating LDA Income Graphs...')

    x = np.arange(1, 2)
    plt.bar(x, income_scores)
    plt.title('Scores for Varying Components in LDA for Income Dataset')
    plt.xlabel('n_components')
    plt.ylabel('score')
    plt.savefig(fname='LDAAnalysisIncome')
    plt.close()

    clf = LinearDiscriminantAnalysis(n_components=1)
    clf.fit(income_X, income_Y)
    plt.bar([1], clf.explained_variance_ratio_)
    plt.title('LDA Explained Variance Ratio for Income Dataset')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.savefig(fname='LDAExplainedVarianceRatioIncome')
    plt.close()

    if verbose:
        print('Done. Beginning LDA Analysis on Music Data...')

    music_score = []
    for n in range(1, 11):
        clf = LinearDiscriminantAnalysis(n_components=n)
        clf.fit(music_x_train, music_y_train)
        music_score.append(clf.score(music_x_test, music_y_test))
    lda_music=clf.fit_transform(music_X, music_Y)

    if verbose:
        print('Done. Creating LDA Music Graphs...')

    x = np.arange(1, 11)
    plt.plot(x, music_score)
    plt.title('Scores for Varying Components in LDA for Music Dataset')
    plt.xlabel('n_components')
    plt.ylabel('score')
    plt.savefig(fname='LDAAnalysisMusic')
    plt.close()

    clf = LinearDiscriminantAnalysis(n_components=4)
    clf.fit(music_X, music_Y)
    plt.bar([1, 2, 3, 4], clf.explained_variance_ratio_)
    plt.title('LDA Explained Variance Ratio for Music Dataset')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.savefig(fname='LDAExplainedVarianceRatioMusic')
    plt.close()

    if verbose:
        print('Reduction Analysis Complete.')

    return pca_income, pca_music, ica_income, ica_music, rp_income, rp_music, lda_income, lda_music

def ReductionClusteringAnalysis(pca_income, pca_music, ica_income, ica_music, rp_income, rp_music, lda_income,
                                lda_music, income_k, music_k, verbose=False):
    train_sizes = np.linspace(.1, 1.0, 10)
    if verbose:
        print('Beginning Reduction + Clustering Analysis. Starting with PCA & KMeans on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(KMeans(
        n_clusters=income_k), pca_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('PCA + KMeans Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='PCAKMeansLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('PCA + KMeans Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='PCAKMeansLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with PCA & Kmeans on Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(KMeans(
        n_clusters=music_k), pca_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('PCA + KMeans Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='PCAKMeansLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('PCA + KMeans Fit Times for Music Dataset')
    plt.xlabel('Training Size')
    plt.ylabel('Fit Time')
    plt.savefig(fname='PCAKMeansLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Reduction + Clustering Analysis with ICA & KMeans on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(KMeans(
        n_clusters=income_k), ica_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('ICA + KMeans Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='ICAKMeansLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('ICA + KMeans Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='ICAKMeansLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with ICA & Kmeans on Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(KMeans(
        n_clusters=music_k), ica_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('ICA + KMeans Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='ICAKMeansLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('ICA + KMeans Fit Times for Music Dataset')
    plt.xlabel('Training Size')
    plt.ylabel('Fit Time')
    plt.savefig(fname='ICAKMeansLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Reduction + Clustering Analysis with RP & KMeans on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(KMeans(
        n_clusters=income_k), rp_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('RP + KMeans Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='RPKMeansLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('RP + KMeans Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='RPKMeansLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with RP & Kmeans on Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(KMeans(
        n_clusters=music_k), rp_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('RP + KMeans Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='RPKMeansLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('RP + KMeans Fit Times for Music Dataset')
    plt.xlabel('Training Size')
    plt.ylabel('Fit Time')
    plt.savefig(fname='RPKMeansLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Reduction + Clustering Analysis with LDA & KMeans on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(KMeans(
        n_clusters=income_k), lda_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('LDA + KMeans Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='LDAKMeansLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('LDA + KMeans Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='LDAKMeansLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with LDA & Kmeans on Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(KMeans(
        n_clusters=music_k), lda_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('LDA + KMeans Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='LDAKMeansLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('LDA + KMeans Fit Times for Music Dataset')
    plt.xlabel('Training Size')
    plt.ylabel('Fit Time')
    plt.savefig(fname='LDAKMeansLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with PCA & EM on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(GaussianMixture(
        n_components=income_k), pca_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('PCA + EM Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='PCAEMLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('PCA + EM Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='PCAEMLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with PCA & EM Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(GaussianMixture(
        n_components=music_k), pca_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('PCA + EM Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='PCAEMLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('PCA + EM Fit Times for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='PCAEMLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with ICA & EM on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(GaussianMixture(
        n_components=income_k), ica_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('ICA + EM Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='ICAEMLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('ICA + EM Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='ICAEMLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with ICA & EM Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(GaussianMixture(
        n_components=music_k), ica_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('ICA + EM Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='ICAEMLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('ICA + EM Fit Times for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='ICAEMLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with RP & EM on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(GaussianMixture(
        n_components=income_k), rp_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    income_train_scores = [x for x in income_train_scores if np.isnan(x) == False]
    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('RP + EM Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='RPEMLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('RP + EM Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='RPEMLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with RP & EM Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(GaussianMixture(
        n_components=music_k), rp_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('RP + EM Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='RPEMLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('RP + EM Fit Times for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='RPEMLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with LDA & EM on Income Dataset...')

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(GaussianMixture(
        n_components=income_k), lda_income, income_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    income_train_scores = np.mean(income_train_scores, axis=1)
    income_valid_scores = np.mean(income_valid_scores, axis=1)
    income_fit_times = np.mean(income_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(income_train_sizes, income_train_scores, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores, label='Validation Scores')
    plt.title('LDA + EM Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='LDAEMLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times)
    plt.title('LDA + EM Fit Times for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='LDAEMLearningCurveTimesIncome')
    plt.close()

    if verbose:
        print('Done. Beginning Analysis with LDA & EM Music Dataset...')

    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(GaussianMixture(
        n_components=music_k), lda_music, music_Y, train_sizes=train_sizes, scoring='accuracy', return_times=True)
    music_train_scores = np.mean(music_train_scores, axis=1)
    music_valid_scores = np.mean(music_valid_scores, axis=1)
    music_fit_times = np.mean(music_fit_times, axis=1)

    if verbose:
        print('Done. Creating Graphs...')

    plt.plot(music_train_sizes, music_train_scores, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores, label='Validation Scores')
    plt.title('LDA + EM Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='LDAEMLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times)
    plt.title('LDA + EM Fit Times for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='LDAEMLearningCurveTimesMusic')
    plt.close()

    if verbose:
        print('Completed Reduction + Clustering Analysis.')

def ReductionNNAnalysis(pca_income, ica_income, rp_income, lda_income, verbose=False):
    train_sizes = np.linspace(.1, 1.0, 10)
    reductions = [pca_income, ica_income, rp_income, lda_income]
    reduction_names = ['PCA', 'ICA', 'RP', 'LDA']

    if verbose:
        print('Beginning Reduction + NN Analysis')

    i = 0
    for X in reductions:
        if verbose:
            print('Beginning Reduction Analysis for NN with ', reduction_names[i])
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(MLPClassifier(
            hidden_layer_sizes=(100, 100, 100, 100, 100), activation='tanh', verbose=verbose), X, income_Y, cv=10,
            train_sizes=train_sizes, return_times=True)
        train_scores = np.mean(train_scores, axis=1)
        test_scores = np.mean(test_scores, axis=1)
        fit_times = np.mean(fit_times, axis=1)

        fnamescore = reduction_names[i] + 'NNLearningCurveIncome'
        fnametimes = reduction_names[i] + 'NNLearningCurveTimesIncome'

        plt.plot(train_sizes, train_scores, label='Training Scores')
        plt.plot(train_sizes, test_scores, label='Validation Scores')
        plt.title(reduction_names[i] + ' + NN Learning Curve for Income Dataset')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.savefig(fname=fnamescore)
        plt.close()

        plt.plot(train_sizes, fit_times)
        plt.title(reduction_names[i] + ' + NN Fit Times for Income Dataset')
        plt.xlabel('Training Size')
        plt.ylabel('Fit Time')
        plt.savefig(fname=fnametimes)
        plt.close()

        i += 1
        if verbose:
            print('Done.')
    if verbose:
        print('Reduction + NN Analysis Complete.')

def ReductionClusteringNNAnalysis(income_Y, pca_income, ica_income, rp_income, lda_income, income_k, verbose=False):
    train_sizes = np.linspace(.1, 1.0, 10)
    reductions = [pca_income, ica_income, rp_income, lda_income]
    reduction_names = ['PCA', 'ICA', 'RP', 'LDA']

    if verbose:
        print('Beginning Reduction + Cluster + NN Analysis...')

    i = 0
    for X in reductions:
        if verbose:
            print('Fitting KMeans + ', reduction_names[i], '...')
        clusters = KMeans(n_clusters=income_k).fit_predict(X).reshape(-1, X.shape[0])
        new_x = np.concatenate((X, clusters.T), axis=1)
        if verbose:
            print('Fitted. Training NN...')

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(MLPClassifier(
            hidden_layer_sizes=(100, 100, 100, 100, 100), activation='tanh'), new_x, income_Y, cv=10,
            train_sizes=train_sizes, return_times=verbose)
        train_scores = np.mean(train_scores, axis=1)
        test_scores = np.mean(test_scores, axis=1)
        fit_times = np.mean(fit_times, axis=1)

        if verbose:
            print('Trained. Creating Graphs...')

        fnamescore = reduction_names[i] + 'KMeansNNLearningCurveIncome'
        fnametimes = reduction_names[i] + 'KMeansNNLearningCurveTimesIncome'

        plt.plot(train_sizes, train_scores, label='Training Scores')
        plt.plot(train_sizes, test_scores, label='Validation Scores')
        plt.title(reduction_names[i] + ' + KMeans + NN Learning Curve for Income Dataset')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.savefig(fname=fnamescore)
        plt.close()

        plt.plot(train_sizes, fit_times)
        plt.title(reduction_names[i] + ' + KMeans + NN Fit Times for Income Dataset')
        plt.xlabel('Training Size')
        plt.ylabel('Fit Time')
        plt.savefig(fname=fnametimes)
        plt.close()

        if verbose:
            print('Fitting EM + ', reduction_names[i], '...')
        if reduction_names[i] == 'RP':
            clusters = GaussianMixture(n_components=income_k, reg_covar=1e-5).fit_predict(X).reshape(-1, X.shape[0])
        else:
            clusters = GaussianMixture(n_components=income_k).fit_predict(X).reshape(-1, X.shape[0])
        new_x = np.concatenate((X, clusters.T), axis=1)
        if verbose:
            print('Fitted. Training NN...')

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(MLPClassifier(
            hidden_layer_sizes=(100, 100, 100, 100, 100), activation='tanh'), new_x, income_Y, cv=10,
            train_sizes=train_sizes, return_times=verbose)
        train_scores = np.mean(train_scores, axis=1)
        test_scores = np.mean(test_scores, axis=1)
        fit_times = np.mean(fit_times, axis=1)

        if verbose:
            print('Trained. Creating Graphs...')

        fnamescore = reduction_names[i] + 'EMNNLearningCurveIncome'
        fnametimes = reduction_names[i] + 'EMNNLearningCurveTimesIncome'

        train_scores = [x for x in train_scores if np.isnan(x) == False]
        test_scores = [x for x in test_scores if np.isnan(x) == False]
        plt.plot(train_sizes, train_scores, label='Training Scores')
        plt.plot(train_sizes, test_scores, label='Validation Scores')
        plt.title(reduction_names[i] + ' + EM + NN Learning Curve for Income Dataset')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.savefig(fname=fnamescore)
        plt.close()

        plt.plot(train_sizes, fit_times)
        plt.title(reduction_names[i] + ' + EM + NN Fit Times for Income Dataset')
        plt.xlabel('Training Size')
        plt.ylabel('Fit Time')
        plt.savefig(fname=fnametimes)
        plt.close()

        i += 1

    if verbose:
        print('Reduction + Cluster + NN Analysis Complete.')

if __name__ == '__main__':
    print('Retrieving Data...')
    income_X, income_Y, music_X, music_Y = get_data()
    print('Data Retrieved.')

    ClusteringTuning(income_X, income_Y, music_X, music_Y, verbose=True)

    income_k = 3
    music_k = 6

    ClusteringAnalysis(income_k, music_k, income_X, income_Y, music_X, music_Y, verbose=True)

    pca_income, pca_music, ica_income, ica_music, rp_income, rp_music, lda_income, lda_music = \
        ReductionAnalysis(income_X, income_Y, music_X, music_Y, verbose=False)

    ReductionClusteringAnalysis(pca_income, pca_music, ica_income, ica_music, rp_income, rp_music, lda_income,
                                lda_music, income_k, music_k, verbose=False)

    ReductionNNAnalysis(pca_income, ica_income, rp_income, lda_income, verbose=True)

    ReductionClusteringNNAnalysis(income_Y, pca_income, ica_income, rp_income, lda_income, income_k, verbose=True)