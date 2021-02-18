% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% Version 1.100
%
% Updated by Computational Cognitive Neuroscience Lab
% University of Padova
% ccnl.psy.unipd.it
%
% Implementation on graphic processors (GPUs) using MATLAB Parallel Computing Toolbox

clear all; close all; clc

% DEEP NETWORK SETUP
% (parameters and final network weights will be saved in structure DN)
DN.layersize   = [100 100 500];        % network architecture
DN.nlayers     = length(DN.layersize);
DN.maxepochs   = 30;                    % unsupervised learning epochs
DN.batchsize   = 300;                   % mini-batch size
sparsity       = 1;                     % set to 1 to encourage sparsity on third layer
spars_factor   = 0.05;                  % how much sparsity?
epsilonw       = 0.1;                   % learning rate (weights)
epsilonvb      = 0.1;                   % learning rate (visible biases)
epsilonhb      = 0.1;                   % learning rate (hidden biases)
weightcost     = 0.0002;                % decay factor
init_momentum  = 0.5;                   % initial momentum coefficient
final_momentum = 0.9;                   % momentum coefficient

% load training dataset
% fname = ['MNIST_data_' sprintf('%d',DN.batchsize) '.mat'];
% load(fname);
load('EMNIST-digits_300.mat')
fprintf(1,'\nUnsupervised training of a deep belief net\n');
DN.err = zeros(DN.maxepochs, DN.nlayers, 'single');
%fflush(stdout); % if running in Octave
tic();

for layer = 1:DN.nlayers
    
    % for the first layer, input data are raw images
    % for next layers, input data are preceding hidden activations
    fprintf(1,'Training layer %d...\n', layer);
    %fflush(stdout); % if running in Octave
    if layer == 1
        data = batchdata;
    else
        data  = batchposhidprobs;
    end
    
    % initialize weights and biases
    numhid  = DN.layersize(layer);
    [numcases, numdims, numbatches] = size(data);
    vishid       = 0.1*randn(numdims, numhid);
    hidbiases    = zeros(1,numhid);
    visbiases    = zeros(1,numdims);
    vishidinc    = zeros(numdims, numhid);
    hidbiasinc   = zeros(1,numhid);
    visbiasinc   = zeros(1,numdims);
    batchposhidprobs = zeros(DN.batchsize, numhid, numbatches);
    
    for epoch = 1:DN.maxepochs
        errsum = 0;
        for mb = 1:numbatches
            data_mb = data(:, :, mb);
            % learn an RBM with 1-step contrastive divergence
            rbm_CPU;
            errsum = errsum + err;
            if epoch == DN.maxepochs
                batchposhidprobs(:, :, mb) = poshidprobs;
            end
            if sparsity && (layer == 3)
                poshidact = sum(poshidprobs);
                Q = poshidact/DN.batchsize;
                if mean(Q) > spars_factor
                    hidbiases = hidbiases - epsilonhb*(Q-spars_factor);
                end
            end
        end
        DN.err(epoch, layer) = errsum;
    end
    % save learned weights
    DN.L{layer}.hidbiases  = hidbiases;
    DN.L{layer}.vishid     = vishid;
    DN.L{layer}.visbiases  = visbiases;
    
end

DN.learningtime = toc();
fprintf(1, '\nElapsed time: %d \n', DN.learningtime);
%fflush(stdout); % if running in Octave
fname = ['DBN_Test' sprintf('%d',DN.batchsize) '.mat'];
% save final network and parameters
save(fname, 'DN');
% if running Octave:
%save('-mat-binary', fname, 'DN');
clear all;

% for i = 1:20
% test = reshape(l.dataset.train.images(i, :), [28 28]);
% imshow(test);
% pause;
% end
% 
% l = load('emnist-digits.mat');
% a = im2double(l.dataset.train.images);
% b = l.dataset.train.labels';
% 
% tar = zeros(length(b), 10);
% for i= 1: length(b)
%     bon = zeros(1,10);
%     bon(b(i)+1) = 1;
%     tar(i,:) = bon;
% end
% 
% a1 = im2double(l.dataset.test.images);
% b1 = l.dataset.test.labels;
% 
% tar1 = zeros(length(b1),10);
% for i= 1: length(b1)
%     bon1 = zeros(1,10);
%     bon1(b1(i)+1) = 1;
%     tar1(i,:) = bon1;
% end
% 
% [r,c] = size(a);
% nlay  = 300;
% batchdata = permute(reshape(a',[c,r/nlay,nlay]),[3,1,2]);
% [r,c] = size(tar);
% batchtargets = permute(reshape(tar',[c,r/nlay,nlay]),[3,1,2]);
% 
% nlay1 = 200;
% [r,c] = size(a1);
% testbatchdata = permute(reshape(a1',[c,r/nlay1,nlay1]),[3,1,2]);
% [r,c] = size(tar1);
% testbatchtargets = permute(reshape(tar1',[c,r/nlay1,nlay1]),[3,1,2]);
% 
% save('sample-emnist-digits.mat','batchdata','batchtargets', 'testbatchdata','testbatchtargets')
% % 
% a = l.dataset.train.images;
% [r,c] = size(a);
% nlay  = 300;
% out   = permute(reshape(a',[c,r/nlay,nlay]),[2,1,3]);
% 
% 
% for j= 1:300
%     batch=[300 800 784];
%     for i= 1:800
%         a = l.dataset.train.images(j*i,:);
% %         b = l.dataset.train.images(j*i,:)
%         batch(j,i,:) = a;
%     end
% end
% 
% 
% figure;
% for i = 1:20
%     subplot(4,5,i);
%     test = reshape(batchdata(i,:, 1), [28 28]);
%     imshow(test);
% end
