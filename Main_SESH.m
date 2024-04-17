function Main_SESH()
clear;
clc;
%% parameters setting
param=getGobalOptions( ... 
        'nbits',[16],...  
        'maxIter',[15],...                     
        'lambda',[8],...           
        'alpha',[1e-2],...       
        'beta',[1e-3],...          
        'mu',[1e-2],...
        'gamma',[1e-5]);
n_anchors=1500;
%% load dataset
% dataname = 'MIRFLICKR';
dataname = 'mirflickr5k';
datasets = ['datasets/' dataname '.mat'];
load(datasets); 
XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
XTest = I_te; YTest = T_te; LTest = L_te;
if ~exist('result/', 'dir')
    mkdir('result/');
end

%% kernel
rbf2;
%% initial data
KX = XKTrain * XKTrain';
KY = YKTrain * YKTrain';
mainfold.Metric = 'Euclidean';
mainfold.NeighborMode = 'KNN';
mainfold.k = 100;
DX = full(lapgraph(XKTrain,mainfold));
DY = full(lapgraph(YKTrain,mainfold));
param.nbits = 16;
param.maxIter = 15;
param.lambda = 8;
param.alpha = 1e-2;
param.beta = 1e-3;
param.mu = 1e-2;
param.gamma = 1e-5;
param.KX = KX;
param.KY = KY;
param.DX = DX;
param.DY = DY;
%% train
param.sampleColumn = 2 * param.nbits;
eva_info=evaluate_SESH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,param);
% MAP
param.Image_to_Text_MAP = eva_info.Image_to_Text_MAP;
param.Text_to_Image_MAP = eva_info.Text_to_Image_MAP;
param.mean_MAP= mean([param.Image_to_Text_MAP, param.Text_to_Image_MAP]);
% train time
param.trainT = eva_info.trainT;
result = rmfield(param, {'KX','KY','DX','DY'});
fprintf('traintime: %f\n',result.trainT);
% nowTime=getNowTime();
% result_path = ['result/result_' dataname nowTime];
% save(result_path,'result');

end

