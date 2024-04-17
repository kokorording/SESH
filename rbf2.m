[n, ~] = size(YTrain);
anchor_image = XTrain(randsample(n, n_anchors),:); 
anchor_text = YTrain(randsample(n, n_anchors),:);
XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 