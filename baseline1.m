function[testError,cvError]=baseline1(trainSet,testSet,kfolds)
    cvp = cvpartition(size(trainSet,1), 'KFold', kfolds);
    valerrorRate=zeros(kfolds,1);
    trainerrorRate=zeros(1000,1);
    w_array=zeros(1000,size(trainSet,2)); %1000*58
    N=10; % number of most significant features used to decide spam
    for i=1:kfolds  
        trainingFeatures=trainSet(cvp.training(i),1:57);
        trainingLabels=trainSet(cvp.training(i),58);
        validationFeatures=trainSet(cvp.test(i),1:57);
        validationLabels=trainSet(cvp.test(i),58); 
        %normalise data
        trainingFeatures=zscore(trainingFeatures);
        validationFeatures=zscore(validationFeatures);
       
        spamIndex=find(trainingLabels==1);
        spamfeaturesSum=sum(trainingFeatures(spamIndex,:));
        spamfeaturesAve=spamfeaturesSum/size(spamIndex,1);
        nonspamIndex= find(trainingLabels==0);
        nonspamfeaturesSum=sum(trainingFeatures(nonspamIndex,:));
        nonspamfeaturesAve=nonspamfeaturesSum./size(nonspamIndex,1);
        featuresDifference=spamfeaturesAve-nonspamfeaturesAve;
      
        [sortedValues,sortIndex]=sort(featuresDifference,'descend'); % sort values in descending order
        maxIndex=sortIndex(1:N); % Obtain index in original array corresponding to largest N values
       
        %Validation
        counter=0;
            for m=1:size(validationFeatures,1)
                if size(find(validationFeatures(m,maxIndex)>0))==size(maxIndex)
                    prediction=1;
                    if prediction~=validationLabels(m)
                       counter=counter+1;
                    end
                else
                    prediction=0;
                    if prediction~=validationLabels(m)
                        counter=counter+1;
                    end
                end
            end
            valerrorRate(i,1)=counter/size(validationFeatures,1);
       end
    cvError=sum(valerrorRate)/kfolds;
    
    %test accuracy
    spamIndex=find(trainSet(:,58)==1);
    spamfeaturesSum=sum(trainSet(spamIndex,1:57));
    spamfeaturesAve=spamfeaturesSum/size(spamIndex,1);
    nonspamIndex= find(trainSet(:,58)==0);
    nonspamfeaturesSum=sum(trainSet(nonspamIndex,1:57));
    nonspamfeaturesAve=nonspamfeaturesSum./size(nonspamIndex,1);
    featuresDifference=spamfeaturesAve-nonspamfeaturesAve;
    [sortedValues,sortIndex]=sort(featuresDifference,'descend'); % sort values in descending order
    maxIndex=sortIndex(1:N); % Obtain index in original array corresponding to largest N values
    
     counter=0;
     for m=1:size(testSet,1)
             if size(find(testSet(m,maxIndex)>0))==size(maxIndex)
                    prediction=1;
                    if prediction~=testSet(m,58)
                       counter=counter+1;
                    end
             else
                    prediction=0;
                    if prediction~=testSet(m,58)
                        counter=counter+1;
                    end
             end
     end
    testError=counter/size(testSet,1);     
end