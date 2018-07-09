close all
clear 
clc

rng('default');
data=importdata('spambase.data');
trainRatio=9;
testRatio=1;
N=size(data,1);
[trainInd,testInd]=dividerand(N,trainRatio,testRatio);
trainData=data(trainInd,:);
testData=data(testInd,:);
trainData=trainData(randperm(size(trainData,1)),:);
testData=testData(randperm(size(testData,1)),:);
feature=data(:,1:57);
label=data(:,58);
k_fold=10;

%baseline 1 with CV
%[testError,cvError]=baseline1(trainData,testData,k_fold);

%baseline 2 with CV
%[testError,cvError]=baseline2(trainData,testData,k_fold);

%Linear regression: Lasso
lambda=0:0.0001:0.5;
trainSet=trainData;
testSet=testData;
trainSet(:,1:57)=zscore(trainSet(:,1:57));
testSet(:,1:57)=zscore(testSet(:,1:57));
trainSet(find(trainSet(:,58)==0),58)=-1;
testSet(find(testSet(:,58)==0),58)=-1;
[B]=lasso(trainSet(:,1:57),trainSet(:,58),'Lambda',lambda,'CV',10);
accuracyrate=zeros(1,size(B,2));
for i=1:size(B,2)
        accuracyrate(1,i)=length(find(sign(testSet(:,1:57)*B(:,i))==testSet(:,58)))/size(testSet,1);
end
[M,I]=max(accuracyrate);
accuracyrate=1-accuracyrate;
lamda_optimal=lambda(I);
plot(lambda,accuracyrate);
title('CV accuracy against regularization parameter');
grid on;
grid minor;
xlabel('Lambda');
ylabel('CV accuracy');
%lassoPlot(B,FitInfo,'PlotType','CV');

%Elastic net
trainSet=trainData;
testSet=testData;
trainSet(:,1:57)=zscore(trainSet(:,1:57));
testSet(:,1:57)=zscore(testSet(:,1:57));
trainSet(find(trainSet(:,58)==0),58)=-1;
testSet(find(testSet(:,58)==0),58)=-1;
[B]=lasso(trainSet(:,1:57),trainSet(:,58),'Lambda',lambda,'Alpha',0.1,'CV',10);
accuracyrate=zeros(1,size(B,2));
for i=1:size(B,2)
        accuracyrate(1,i)=length(find(sign(testSet(:,1:57)*B(:,i))==testSet(:,58)))/size(testSet,1);
end
[M,I]=max(accuracy);
%indices=crossvalind('kfold',4080,10);


%logistic regression
% [trainedModel, cvAccuracy] = logistic_regression(trainData);
% yfit=trainedModel.predictFcn(testData(:,1:57));
% testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);

%SVM Linear Kernel
[trainedModel, cvAccuracy] =svm_linear(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);
svm=trainedModel.ClassificationSVM.IsSupportVector==1;
svmcount=length(find(trainedModel.ClassificationSVM.IsSupportVector==1));

%SVM Quadratic Kernel
[trainedModel, cvAccuracy] =svm_quadratic(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);
svm=trainedModel.ClassificationSVM.IsSupportVector==1;
svmcount=length(find(trainedModel.ClassificationSVM.IsSupportVector==1));

%SVM Cubic Kernel
[trainedModel, cvAccuracy] =svm_cubic(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);
svm=trainedModel.ClassificationSVM.IsSupportVector==1;
svmcount=length(find(trainedModel.ClassificationSVM.IsSupportVector==1));

%SVM Fine RBF Kernel
[trainedModel, cvAccuracy] =svm_fine_rbf(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);
svm=trainedModel.ClassificationSVM.IsSupportVector==1;
svmcount=length(find(trainedModel.ClassificationSVM.IsSupportVector==1));

%SVM Medium RBF Kernel
[trainedModel, cvAccuracy] =svm_medium_rbf(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);
svm=trainedModel.ClassificationSVM.IsSupportVector==1;
svmcount=length(find(trainedModel.ClassificationSVM.IsSupportVector==1));

%SVM Coarse RBF Kernel
[trainedModel, cvAccuracy] =svm_coarse_rbf(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);
svm=trainedModel.ClassificationSVM.IsSupportVector==1;
svmcount=length(find(trainedModel.ClassificationSVM.IsSupportVector==1));

%KNN,K=1
[trainedModel, cvAccuracy] =knn_1(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);

%KNN,K=10
[trainedModel, cvAccuracy] =knn_10(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);

%KNN,K=100
[trainedModel, cvAccuracy] =knn_100(trainData);
yfit=trainedModel.predictFcn(testData(:,1:57));
testAccuracy=length(find(testData(:,58)==yfit))/size(testData,1);

%Neuro Network
% feature=zscore(feature);
% label=zscore(label);
% [trainAccuracy,cvAccuracy,testPerformance]=neuro_net(feature,label);