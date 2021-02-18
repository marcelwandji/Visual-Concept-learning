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

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)

% Version 1.100
%
% Updated by Computational Cognitive Neuroscience Lab
% University of Padova
% ccnl.psy.unipd.it
%
% Implementation on graphic processors (GPUs) using MATLAB Parallel Computing Toolbox


momentum = init_momentum;

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%
poshidprobs  = 1./(1 + exp(-data_mb * vishid - repmat(hidbiases, numcases, 1)));
posprods     = data_mb' * poshidprobs; 
poshidact    = sum(poshidprobs); 
posvisact    = sum(data_mb);
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%
poshidstates = poshidprobs > rand(numcases, numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%
negdata     = 1./(1 + exp(-poshidstates * vishid' - repmat(visbiases, numcases, 1))); 
neghidprobs = 1./(1 + exp(-negdata * vishid       - repmat(hidbiases, numcases, 1)));
negprods    = negdata' * neghidprobs;
neghidact   = sum(neghidprobs);
negvisact   = sum(negdata);
%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%

err = sqrt(sum(sum((data_mb - negdata).^2)));
if epoch > 5,
    momentum = final_momentum;
end

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%
vishidinc  = momentum * vishidinc  + epsilonw*( (posprods-negprods)/numcases - weightcost * vishid);
visbiasinc = momentum * visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
hidbiasinc = momentum * hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
vishid     = vishid + vishidinc;
visbiases  = visbiases + visbiasinc;
hidbiases  = hidbiases + hidbiasinc;
%%%%%%%%% END OF UPDATES %%%%%%%%%
