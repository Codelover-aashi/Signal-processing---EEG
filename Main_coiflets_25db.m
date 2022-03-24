% noise 25 db
% dwt-coiflets

%Clear command window
clc
%CLEAR removes all variables from the workspace
clear all
%Close  all figure from the workspace
close all

[file path]=uigetfile(['EEG Data\']);
load([path file])
data=o.data';
data=data(4:17,:);

snr_val=[25];
data_noise=awgn(data,1,snr_val);
[Pre_processed_data,W] = Preprocessing(data_noise);
final_pre_data = W*Pre_processed_data;
% Plotting 9 th Preprocessed signal only
data1=final_pre_data(:,1:2000);
data2=final_pre_data(:,100001:102000);
data3=final_pre_data(:,200001:202000);
final_pre_data_train=[data1 data2 data3];
%%%%%%%%%%% features Extraction %%%%%%%%%%%%%%%%
wlen = 8;                        % window length (recomended to be power of 2)
hop = wlen/2;                       % hop size (recomended to be power of 2)
nfft = 512;                        % number of fft points (recomended to be power of 2)
% perform STFT
win = blackman(wlen, 'periodic');
for ii=1:6000
    ii
    [S, f, t] = stft(final_pre_data_train(:,ii), win, hop, nfft, 64);
    % calculate the coherent amplification of the window
    C = sum(win)/wlen;
    
    % take the amplitude of fft(x) and scale it, so not to be a
    % function of the length of the window and its coherent amplification
    S = abs(S)/wlen/C;
    
    % correction of the DC & Nyquist component
    if rem(nfft, 2)                     % odd nfft excludes Nyquist point
        S(2:end, :) = S(2:end, :).*2;
    else                                % even nfft includes Nyquist point
        S(2:end-1, :) = S(2:end-1, :).*2;
    end
    
    % convert amplitude spectrum to dB (min = -120 dB)
    S = 20*log10(S + 1e-6);
    [ca,cd] = dwt(final_pre_data_train(:,1),'coif1','mode','sym');
    fea_tures=[];
    fea_tures=[ca;cd;S(:)];
    feature_train(ii,:)=fea_tures;
end

data_1=final_pre_data(:,1501:2500);
data_2=final_pre_data(:,101501:102500);
data_3=final_pre_data(:,201501:202500);
final_pre_data_test=[data_1 data_2 data_3];

for ii=1:3000
    ii
    [S, f, t] = stft(final_pre_data_test(:,ii), win, hop, nfft, 128);
    % calculate the coherent amplification of the window
    C = sum(win)/wlen;
    
    % take the amplitude of fft(x) and scale it, so not to be a
    % function of the length of the window and its coherent amplification
    S = abs(S)/wlen/C;
    
    % correction of the DC & Nyquist component
    if rem(nfft, 2)                     % odd nfft excludes Nyquist point
        S(2:end, :) = S(2:end, :).*2;
    else                                % even nfft includes Nyquist point
        S(2:end-1, :) = S(2:end-1, :).*2;
    end
    
    % convert amplitude spectrum to dB (min = -120 dB)
    S = 20*log10(S + 1e-6);
    [ca,cd] = dwt(final_pre_data_train(:,1),'coif1','mode','sym');
    fea_tures=[];
    fea_tures=[ca;cd;S(:)];
    feature_test(ii,:)=fea_tures;
end
new_test=feature_test;
load label
load label_test
feature_train=[feature_train;feature_train];
label=[label;label];
label_test=categorical(label_test)';
label_train=categorical(label);
% decision tree
net=fitctree(feature_train,label_train);
pre_dict_train=predict(net,feature_train);
pre_dict_test=predict(net,new_test);
%train
stats=[];
stats = Performance_measure(double(label_train),double(pre_dict_train));
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_tree_train=[ accuracy sensitivity specificity precision Fscore];
%test
stats=[];
stats = Performance_measure(double(label_test),double(pre_dict_test));
roc_curve(double(label_test)',double(pre_dict_test))
hold on
title('ROC for Desicion Tree Testing');
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_tree_test=[ accuracy sensitivity specificity precision Fscore];

%svm
net=fitcecoc(feature_train,label_train');
pre_svm_train=predict(net,feature_train);
pre_svm_test=predict(net,new_test);

%train
stats=[];
stats = Performance_measure(double(label_train),double(pre_svm_train));
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_svm_train=[ accuracy sensitivity specificity precision Fscore];

% test
stats=[];
stats = Performance_measure(double(label_test),double(pre_svm_test));
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_svm_test=[ accuracy sensitivity specificity precision Fscore];

% Naive bayas
net=fitcnb(feature_train,label_train');
pre_nb_train=predict(net,feature_train);
pre_nb_test=predict(net,new_test);
%train
stats=[];
stats = Performance_measure(double(label_train),double(pre_nb_train));
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_nb_train=[ accuracy sensitivity specificity precision Fscore];

%test
stats=[];
stats = Performance_measure(double(label_test),double(pre_nb_test));
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_nb_test=[ accuracy sensitivity specificity precision Fscore];

% random forest
BaggedEnsemble = generic_random_forests(feature_train,label_train',60,'classification');
pre_rf_train=predict(BaggedEnsemble,feature_train);
for ii=1:length(pre_rf_train)
    rf_pre_train(ii)=str2num(pre_rf_train{ii});
end

pre_rf_test=predict(BaggedEnsemble,new_test);
for ii=1:length(pre_rf_test)
    rf_pre_test(ii)=str2num(pre_rf_test{ii});
end

%train
stats=[];
stats = Performance_measure(double(label_train),double(rf_pre_train));
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_rf_train=[ accuracy sensitivity specificity precision Fscore];
%test
stats=[];
stats = Performance_measure(double(label_test),double(rf_pre_test));
accuracy=mean(stats.accuracy);
sensitivity=mean(stats.sensitivity);
specificity=mean(stats.specificity);
precision=mean(stats.precision);
recall=mean(stats.recall);
Fscore=mean(stats.Fscore);
EVAL_rf_test=[ accuracy sensitivity specificity precision Fscore];


figure
name = {'Accuracy';'Sensitivity';'Specificity';'Precision';'F-Measure'};
x = [1:5];
res=[EVAL_tree_train;EVAL_svm_train;EVAL_nb_train;EVAL_rf_train]*100;
bar(x,res')
set(gca,'xticklabel',name)
legend({'Desicion tree','SVM','Naive Bayes','Random forest'})
title('Performance Analysis Training')

figure
name = {'Accuracy';'Sensitivity';'Specificity';'Precision';'F-Measure'};
x = [1:5];
res=[EVAL_tree_test;EVAL_svm_test;EVAL_nb_test;EVAL_rf_test]*100;
bar(x,res')
set(gca,'xticklabel',name)
legend({'Desicion tree','SVM','Naive Bayes','Random forest'})
title('Performance Analysis Training')

figure
roc_curve(double(label_train),double(pre_dict_train));
hold on
title('ROC for Desicion Tree Training');
figure
roc_curve(double(label_train),double(pre_svm_train));
hold on
title('ROC for SVM Training');
figure
roc_curve(double(label_train),double(pre_nb_train));
hold on
title('ROC for Naive Bayes Training');
figure
roc_curve(double(label_train),double(rf_pre_train));
hold on
title('ROC for Random Forest Training');


figure
roc_curve(double(label_test),double(pre_dict_test));
hold on
title('ROC for Desicion Tree Testing');
figure
roc_curve(double(label_test),double(pre_svm_test));
hold on
title('ROC for SVM Testing');
figure
roc_curve(double(label_test),double(pre_nb_test));
hold on
title('ROC for Naive Bayes Testing');
figure
roc_curve(double(label_test),double(rf_pre_test));
hold on
title('ROC for Random Forest Testing');
