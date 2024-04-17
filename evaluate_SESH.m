function evaluation_info = evaluate_SESH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,param)
% training procedure
tic;
[B,Wx,Wy] = train_SESH(XKTrain, YKTrain, LTrain, param);
fprintf('...training finishes\n');
tBX = compactbit(XKTest * Wx > 0);
tBY = compactbit(YKTest * Wy > 0);
dB = compactbit(B > 0);
    traintime=toc;
    evaluation_info.trainT=traintime;
    %% evaluate
    fprintf('evaluating...\n');
    DHamm = hammingDist(tBX, dB);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
    param.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
    fprintf('%dbits Image_to_Text_MAP: %f.\n', param.nbits, evaluation_info.Image_to_Text_MAP);


    DHamm = hammingDist(tBY, dB);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);
    param.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);
    fprintf('%dbits Text_to_Image_MAP: %f.\n', param.nbits, evaluation_info.Text_to_Image_MAP);

end