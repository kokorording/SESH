function [B,Wx,Wy] = train_SESH(XTrain, YTrain, LTrain, param)
eps=0.1;
bit = param.nbits;
maxIter = param.maxIter;
sampleColumn = param.sampleColumn;
lambda = param.lambda;
alpha = param.alpha;
beta = param.beta;
mu = param.mu;
gamma = param.gamma;
KX = param.KX;
KY = param.KY;
DX = param.DX;
DY = param.DY;
[numTrain, dX] = size(XTrain); 
dY = size(YTrain, 2); 
V = ones(numTrain, bit);
V(randn(numTrain, bit) < 0) = -1;

B = ones(numTrain, bit);
B(randn(numTrain, bit) < 0) = -1;

G = randn(numTrain,bit);

for epoch = 1:maxIter
     % sample Sc
    Sc = randperm(numTrain, sampleColumn);
    KXX = KX(:,Sc);
    KYY = KY(:,Sc);
    DXX = DX(:,Sc);
    DYY = DY(:,Sc);

    % update WX
    WX = ((2 * alpha * KXX ) \ (2 * alpha * KXX * G(Sc,:) - mu * DXX * G(Sc,:) )) ;

    % update WY
    WY = ((2 * alpha * KYY ) \ (2 * alpha * KYY * G(Sc,:) - mu * DYY * G(Sc,:) )) ;

    % update G
    P1 = -2 * alpha * KXX * WX + mu * DXX * WX -2 * alpha * KYY* WY + mu * DYY * WY;
    if any(isnan(P1 - 2 * gamma * V), 'all')
        disp('矩阵包含NaN值');
        continue;
    end
    [H1 ,~ ,H2] = svd((P1 - 2 * gamma * V),'econ');
    G= H1  * H2';
    
    % update R1
    R1 = (XTrain' * XTrain + eps*eye(dX)) \ (XTrain' * V);

    % update R2
    R2 = (YTrain' * YTrain + eps*eye(dY)) \ (YTrain' * V);

    % update V
    SX = LTrain * LTrain(Sc, :)' > 0;
    V = updateColumnV(V, B, SX, Sc, bit, lambda, sampleColumn, gamma, G, beta, XTrain, YTrain, R1, R2);

    % update B
    SY = LTrain(Sc, :) * LTrain' > 0;
    B = updateColumnB(B, V, SY, Sc, bit, lambda, sampleColumn);
end
Wx = R1;
Wy = R2;
end

function V = updateColumnV(V, B, S, Sc, bit, lambda, sampleColumn, gamma, G, beta, XTrain, YTrain, R1, R2)
m = sampleColumn;
n = size(V, 1);
for k = 1: bit
    TX = lambda * V * B(Sc, :)' / bit;
    AX = 1 ./ (1 + exp(-TX));
    Vjk = B(Sc, k)';
    XR1 = XTrain * R1;
    YR2 = YTrain * R2;
    p = lambda * ((S - AX) .* repmat(Vjk, n, 1)) * ones(m, 1) / bit...
        + (m * lambda^2 + 8 * bit^2 *  (2 * beta + gamma))* V(:, k) / (4 * bit^2)...
        + 2 * gamma * (V(:, k) - G(:, k)) + 2 * beta * (B(:, k) - XR1(:, k)) + 2 * beta * (B(:, k) - YR2(:, k));
    V_opt = ones(n, 1);
    V_opt(p < 0) = -1;
    V(:, k) = V_opt;
end
end

function B = updateColumnB(B, V, S, Sc, bit, lambda, sampleColumn)
m = sampleColumn;
n = size(V, 1);
for k = 1: bit
    TX = lambda * V(Sc, :) * B' / bit;
    AX = 1 ./ (1 + exp(-TX));
    Bjk = V(Sc, k)';
    p = lambda * ((S' - AX') .* repmat(Bjk, n, 1)) * ones(m, 1)  / bit...
        + (m * lambda^2 )* B(:, k) / (4 * bit^2);
    B_opt = ones(n, 1);
    B_opt(p < 0) = -1;
    B(:, k) = B_opt;
end

end
