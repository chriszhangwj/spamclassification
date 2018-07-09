function[testError,cvError]=baseline2(trainSet,testSet,kfolds)
    cvp = cvpartition(size(trainSet,1), 'KFold', kfolds);
    trainSet(find(trainSet(:,58)==0),58)=-1;
    %zeroindTrain=find(trainSet(:,58)==0);
   % trainSet(zeroindTrain,58)=-1;
    %trainSet(find(trainSet(:,58)=0),:)=0;
    testSet(find(testSet(:,58)==0),58)=-1;
    testSet=[zeros(size(testSet,1),1) testSet];
    valerrorRate=zeros(kfolds,1);
    trainerrorRate=zeros(1000,1);
    w_array=zeros(1000,size(trainSet,2)); %1000*58
    for i=1:kfolds  % ith iteration for CV
       trainingFeatures=trainSet(cvp.training(i),1:57);
       trainingLabels=trainSet(cvp.training(i),58);
       validationFeatures=trainSet(cvp.test(i),1:57);
       validationLabels=trainSet(cvp.test(i),58); 
       trainingFeatures=[zeros(size(trainingFeatures,1),1) trainingFeatures];
       validationFeatures=[zeros(size(validationFeatures,1),1) validationFeatures];
       
       %perceptron
       n=size(trainingFeatures,1);
       w=zeros(1,58);
       for iteration=1:1000
            for j=1:n
                x=trainingFeatures(j,:);
                predictLabels(j)=sign(w*(x.'));
            end
            errorIndex=find(predictLabels~=trainingLabels.');
            pickIndex=errorIndex(randi(length(errorIndex)));    
            w=w+trainingLabels(pickIndex)*trainingFeatures(pickIndex,:); 
            %pocket
            counter=0;
            for p=1:size(trainingFeatures,1)
                if sign(w*trainingFeatures(p,:).')~=trainingLabels(p,:)
                counter=counter+1;
                end
            end
            trainerrorRate(iteration,1)=counter/size(trainingFeatures,1);
            w_array(iteration,:)=w;
       end
       [M,I]=min(trainerrorRate);
       w=w_array(I,:);
       %validation
        counter=0;
        for m=1:size(validationFeatures,1)
            if sign(w*validationFeatures(m,:).')~=validationLabels(m,:)
                counter=counter+1;
            end
        end
        valerrorRate(i,1)=counter/size(validationFeatures,1);
    end
    cvError=sum(valerrorRate)/kfolds;
 
    %train Accuracy
    counter=0;
    for i=1:size(testSet,1)
        if(sign(w*testSet(i,1:58).')~=testSet(i,59))
           counter=counter+1;
        end
    end
    testError=counter/size(testSet,1);
end
    
    
    