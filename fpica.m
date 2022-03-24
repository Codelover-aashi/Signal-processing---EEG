function [A, W] = fpica(X, whiteningMatrix, dewhiteningMatrix, approach, ...
    numOfIC, g, finetune, a1, a2, myy, stabilization, ...
    epsilon, maxNumIterations, maxFinetune, initState, ...
    guess, sampleSize, displayMode, displayInterval, ...
    s_verbose);

global g_FastICA_interrupt;
if isempty(g_FastICA_interrupt)
    clear global g_FastICA_interrupt;
    interruptible = 0;
else
    interruptible = 1;
end

if nargin < 3, error('Not enough arguments!'); end
[vectorSize, numSamples] = size(X);
if nargin < 20, s_verbose = 'on'; end
if nargin < 19, displayInterval = 1; end
if nargin < 18, displayMode = 'on'; end
if nargin < 17, sampleSize = 1; end
if nargin < 16, guess = 1; end
if nargin < 15, initState = 'rand'; end
if nargin < 14, maxFinetune = 100; end
if nargin < 13, maxNumIterations = 1000; end
if nargin < 12, epsilon = 0.0001; end
if nargin < 11, stabilization = 'on'; end
if nargin < 10, myy = 1; end
if nargin < 9, a2 = 1; end
if nargin < 8, a1 = 1; end
if nargin < 7, finetune = 'off'; end
if nargin < 6, g = 'pow3'; end
if nargin < 5, numOfIC = vectorSize; end     % vectorSize = Dim
if nargin < 4, approach = 'defl'; end
% s_verbose
% switch lower(s_verbose)
%     case 'on'
b_verbose = 1;

approachMode = 2;

if b_verbose, fprintf('Used approach [ %s ].\n', approach); end


if sampleSize > 1
    sampleSize = 1;
    if b_verbose
        fprintf('Warning: Setting ''sampleSize'' to 1.\n');
    end
elseif sampleSize < 1
    if (sampleSize * numSamples) < 1000
        sampleSize = min(1000/numSamples, 1);
        if b_verbose
            fprintf('Warning: Setting ''sampleSize'' to %0.3f (%d samples).\n', ...
                sampleSize, floor(sampleSize * numSamples));
        end
    end
end
if b_verbose
    if  b_verbose & (sampleSize < 1)
        fprintf('Using about %0.0f%% of the samples in random order in every step.\n',sampleSize*100);
    end
end

gOrig = 10;

if sampleSize ~= 1
    gOrig = gOrig + 2;
end
if myy ~= 1
    gOrig = gOrig + 1;
end

if b_verbose,
    fprintf('Used nonlinearity [ %s ].\n', g);
end
finetuningEnabled = 1;
if myy ~= 1
    gFine = gOrig;
else
    gFine = gOrig + 1;
end
finetuningEnabled = 0;

if b_verbose & finetuningEnabled
    fprintf('Finetuning enabled (nonlinearity: [ %s ]).\n', finetune);
end

if myy ~= 1
    stabilizationEnabled = 1;
else
    stabilizationEnabled = 0;
end

if b_verbose & stabilizationEnabled
    fprintf('Using stabilized algorithm.\n');
end
myyOrig = myy;
myyK = 0.01;
failureLimit = 5;
usedNlinearity = gOrig;
stroke = 0;
notFine = 1;
long = 0;
initialStateMode = 0;
usedDisplay = 0;
if displayInterval < 1
    displayInterval = 1;
end
if b_verbose, fprintf('Starting ICA calculation...\n'); end

if approachMode == 2
    B = zeros(vectorSize);
    round = 1;
    numFailures = 0;
    while round <= numOfIC,
        myy = myyOrig;
        usedNlinearity = gOrig;
        stroke = 0;
        notFine = 1;
        long = 0;
        endFinetuning = 0;
        if b_verbose, fprintf('IC %d ', round); end
        if initialStateMode == 0
            w = randn (vectorSize, 1);
        elseif initialStateMode == 1
            w=whiteningMatrix*guess(:,round);
        end
        w = w - B * B' * w;
        w = w / norm(w);
        wOld = zeros(size(w));
        wOld2 = zeros(size(w));
        i = 1;
        gabba = 1;
        while i <= maxNumIterations + gabba
            if (usedDisplay > 0)
                drawnow;
            end
            if (interruptible & g_FastICA_interrupt)
                if b_verbose
                    fprintf('\n\nCalculation interrupted by the user\n');
                end
                return;
            end
            w = w - B * B' * w;
            w = w / norm(w);
            if notFine
                if i == maxNumIterations + 1
                    if b_verbose
                        fprintf('\nComponent number %d did not converge in %d iterations.\n', round, maxNumIterations);
                    end
                    round = round - 1;
                    numFailures = numFailures + 1;
                    if numFailures > failureLimit
                        if b_verbose
                            fprintf('Too many failures to converge (%d). Giving up.\n', numFailures);
                        end
                        if round == 0
                            A=[];
                            W=[];
                        end
                        return;
                    end
                    %
                    break;
                end
            else
                if i >= endFinetuning
                    wOld = w;
                end
            end
            if b_verbose, fprintf('.'); end;
            if norm(w - wOld) < epsilon | norm(w + wOld) < epsilon
                if finetuningEnabled & notFine
                    if b_verbose, fprintf('Initial convergence, fine-tuning: '); end;
                    notFine = 0;
                    gabba = maxFinetune;
                    wOld = zeros(size(w));
                    wOld2 = zeros(size(w));
                    usedNlinearity = gFine;
                    myy = myyK * myyOrig;
                    endFinetuning = maxFinetune + i;
                else
                    numFailures = 0;
                    B(:, round) = w;
                    % Calculate the de-whitened vector.
                    A(:,round) = dewhiteningMatrix * w;
                    % Calculate ICA filter.
                    W(round,:) = w' * whiteningMatrix;
                    if b_verbose, fprintf('computed ( %d steps ) \n', i); end
                    break; % IC ready - next...
                end
            end
            wOld2 = wOld;
            wOld = w;
            % weight updation
            
            w = (X * ((X' * w) .^ 3)) / numSamples - 3 * w;
            % Normalize the new w.
            w = w / norm(w);
            i = i + 1;
        end
        round = round + 1;
    end
    if b_verbose, fprintf('Done.\n'); end
end
if ~isreal(A)
    if b_verbose, fprintf('Warning: removing the imaginary part from the result.\n'); end
    A = real(A);
    W = real(W);
end

function y=tanh(x)
y = 1 - 2 ./ (exp(2 * x) + 1);

function Samples = getSamples(max, percentage)
Samples = find(rand(1, max) < percentage);
