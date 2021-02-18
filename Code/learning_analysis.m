clc; clear;

load('EMNIST-digits_300.mat'); % load training and testing dataset
load('DBN_Test300.mat');        % load pre-trained DBN

%% perform linear read-out at different layers of the hierarchy
fprintf('Reshaping 3D matrixes into 2d ones...\n');

tr_patt = num2cell(batchdata, [1 2]);     % split into cell array keeping dimensions 1 and 2 together
tr_patt = vertcat(tr_patt{:});            % concatenate all the cells vertically
tr_labels = num2cell(batchtargets, [1 2]);% perform the same operation for label vectors
tr_labels = vertcat(tr_labels{:});

te_patt = num2cell(testbatchdata, [1 2]); % split into cell array keeping dimensions 1 and 2 together
te_patt = vertcat(te_patt{:});            % concatenate all the cells vertically
%te_patt = te_patt+rand(40000, 784).*te_patt/0.001; %add noise to the testing dataset
te_labels = num2cell(testbatchtargets, [1 2]);
te_labels = vertcat(te_labels{:});

fprintf('\nGive as input to the classifier the raw images...\n');
[W0, tr_acc0, te_acc0] = perceptron(tr_patt, tr_labels, te_patt, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc0, te_acc0);

fprintf('\nGive as input to the classifier the hidden activations of layer 1..\n');
H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));
H1_te = 1./(1 + exp(-te_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(te_patt,1),1)));
[W1, tr_acc1, te_acc1] = perceptron(H1_tr, tr_labels, H1_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc1, te_acc1);

fprintf('\nGive as input to the classifier the hidden activations of layer 2..\n');
H2_tr = 1./(1 + exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1)));
H2_te = 1./(1 + exp(-H1_te*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_te,1),1)));
[W2, tr_acc2, te_acc2] = perceptron(H2_tr, tr_labels, H2_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc2, te_acc2);

fprintf('\nGive as input to the classifier the hidden activations of layer 3..\n');
H3_tr = 1./(1 + exp(-H2_tr*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_tr,1),1)));
H3_te = 1./(1 + exp(-H2_te*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_te,1),1)));
[W3, tr_acc3, te_acc3] = perceptron(H3_tr, tr_labels, H3_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc3, te_acc3);

%% plot classification test accuracy for each layer
figure();
bar([te_acc0 te_acc1 te_acc2 te_acc3])
ylim([0.8 1]);
ylabel('Test accuracy')
xticklabels({'Pixels', 'H1', 'H2', 'H3'})


% ONES = ones(size(H3_te, 1), 1);
%     te_patterns = [H3_te ONES];
%     pred = te_patterns*W3;
%     [~, max_act] = max(pred,[],2);
% [r,~] = find(te_labels');
% cm = confusionmat(r,max_act);
% confusionchart(cm);
