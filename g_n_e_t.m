%% Foldering
imagesDir='PlantVillage';
originalFileLocation = fullfile(imagesDir);

%%
imds = imageDatastore(originalFileLocation, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
inputSize = [224 224];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
%%
[imdsnew , imdsrest ] = splitEachLabel(imds,0.2,'randomized');
%%
imdsnew = shuffle(imdsnew)

%%
[imdsTrain, imdsval, imdsTest] = splitEachLabel(imdsnew,0.7,0.2,'randomized')
%%
XValidation=imdsval.Files;
YValidation=imdsval.Labels;
%%
net = googlenet;

%%
%%Replace Final Layers

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

learnableLayer=net.Layers(end-2);
classLayer=net.Layers(end);
numClasses = numel(categories(imdsTrain.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%%
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

% pixelRange = [-30 30];
% scaleRange = [0.9 1.1];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange, ...
%     'RandXScale',scaleRange, ...
%     'RandYScale',scaleRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
%%
miniBatchSize = 10;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsval, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%
net = trainNetwork(imdsTrain,lgraph,options);

%%
save net
%%
[YPred,probs] = classify(net,imdsTest);
accuracy = mean(YPred == imdsTest.Labels)

confusionmat(imdsTest.Labels,YPred)

%%
idx = randperm(numel(imdsTest.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%%
% for i=1200:1205
%     img{i}=readimage(imdsnew,i);
%     subplot(3,2,i-1199)
%     imshow(img{i})
%     label = imdsnew.Labels(i);
%     title(string(label));
%     
% end






