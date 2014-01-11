function [model, info] = do_m5il_solve( instDataMat, indBagVec, bagLabel_binary, num_clust, M5IL, DEBUG, VIS )
% M5ILBINCLS_V1: train model of maximum margin multi-modality multiple instance learning (m5il) for binary classification;
%==========================================================================
%   Input_Args:
%          <instDataMat>: data matrix of all instances (~ [num_inst, dim]);
%          <indBagVec>: a vector indicating instance affiliation (~ [num_inst, 1]);
%          <bagLabel_binary>: a vector indicating labels of bags (~ [num_bag, 1]);
%          <M5IL>: options on m5il;
%          
%   Output_Args:
%          <model>: learned model;
%          <info>: information on learning (~ [num_inst, 1]);
%
%   Cross-Reference Information:
%       Called by {buildAtbtByM5IL_run_v1};
%
%       Calling:
%
%   v1:  initial version, created on 2012/11/01 by Jun Zhu (v-junzhu);  
%   v1b: simply the code of iterations in cutting-plane algorithm, modified on 2012/11/03 by Jun Zhu (v-junzhu); 
%   v1c: support to select multiple initial model;
%   v2b: only consider positive instances for constraint (C), modified on 2012/11/10 by Jun Zhu (v-junzhu);
%   v2d: show the sample ratio of major modality for positive instances in positive bags;
%   @ MSRA
%==========================================================================
%   v2b -> v3
%
%==========================================================================

if ~exist('DEBUG', 'var')
    DEBUG = false;
end
if ~exist('VIS', 'var')
    VIS = true;
end


%% 
dim_feat = size(instDataMat,2);
num_inst = size(instDataMat,1);
if length(indBagVec) ~= num_inst
    error('Error: the total number of instances does not match;\n');
end

class_data = class(instDataMat);

%% split into positive and negative;
indicator_pos = (bagLabel_binary(indBagVec) == 1);
% for positive class
indBagVec_pos = indBagVec(indicator_pos);   
instDataMat_pos = instDataMat(indicator_pos,:);
list_indBag_pos = unique(indBagVec_pos);
num_bag_pos = length(list_indBag_pos);

% for negative class
indBagVec_neg = indBagVec(~indicator_pos);  
instDataMat = instDataMat(~indicator_pos,:);
list_indBag_neg = unique(indBagVec_neg);
num_bag_neg = length(list_indBag_neg);

%% initialize model;
switch M5IL.init.type
    case 'spec'
        if iscell(M5IL.init.wInit)
            W_init = M5IL.init.wInit{M5IL.init.ind_spec};    % get_init_model_spec( M5IL.init.ind_spec, M5IL.init.wInit, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat, list_indBag_neg, M5IL );
        else
            W_init = M5IL.init.wInit;
        end
    case 'auto'
        if iscell(M5IL.init.wInit)
            W_init = get_init_model_auto( M5IL.init.wInit, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat, list_indBag_neg, M5IL );
        else
            W_init = M5IL.init.wInit;
        end        
    otherwise
        error('Error: unknown method of initialization;\n');
end        

if isempty(W_init)
    W_init = zeros( [dim_feat num_clust], class_data );
else
    if size(W_init,2)~=num_clust
        error('Error: the number of clusters does not match;\n');
    end
    switch class_data
        case 'double'
            W_init = double(W_init);
        case 'single'
            W_init = single(W_init);
        otherwise
            error('Error: unknown type of data class;\n');
    end
end


%% initialze variables and constraints;
num_const_cls_balance = num_clust^2;
kerMat_cls_balance = zeros([num_const_cls_balance num_const_cls_balance], class_data);
% bVec_cls_balance = zeros([num_const_cls_balance 1]);
uMat_cls_balance = zeros([dim_feat*num_clust num_const_cls_balance], class_data);
ii_const_cls_balance = 0;
for p = 1: num_clust
    for q = 1: num_clust
        ii_const_cls_balance = ii_const_cls_balance + 1; % num_clust*(p-1)+q;
        uMat_cls_balance(:,ii_const_cls_balance) = compt_const_balance_class( p, q, indBagVec_pos, instDataMat_pos, logical(ones(size(indBagVec_pos))), ...
                                                                                     list_indBag_pos, num_bag_pos, num_clust, dim_feat, class_data );
        temp_kVec_ws_cls_balance = uMat_cls_balance(:,1:ii_const_cls_balance)'*uMat_cls_balance(:,ii_const_cls_balance);
        kerMat_cls_balance(ii_const_cls_balance,1:ii_const_cls_balance) = temp_kVec_ws_cls_balance';
        kerMat_cls_balance(1:ii_const_cls_balance,ii_const_cls_balance) = temp_kVec_ws_cls_balance;
%         bVec_cls_balance(ii_const_cls_balance) = -zeta;
    end % q;
end % p;

%% CCCP iterations;
all_loss_ori = [];

all_loss_ub_full = {};
all_loss_ub_end = [];

if VIS
    fprintf('\nStart training (alpha = %0.1f, beta = %0.1f, gamma = %0.1f, zeta = %0.1f):', M5IL.alpha, M5IL.beta, M5IL.gamma, M5IL.zeta);
end

W = W_init;
[loss_ori, slack_ori_mil_pos, slack_ori_mil_neg, slack_ori_mic_pos, loss_reg, margin_cls_balance, r_pos_clust, r_pos_tot] = ...
      compt_loss_ori( W, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat, list_indBag_neg, M5IL, false );
all_loss_ori = [all_loss_ori; loss_ori];

if VIS
    fprintf('\n CCCP_iter %i: loss_ori = %0.2f;\n', 0, all_loss_ori(end));
    fprintf(' ori:  slack_mil_pos = %.2f, slack_mil_neg = %.2f, slack_mic_pos = %.2f, loss_reg = %.2f, margin_cls_balance = %.2f, r_pos_tot = %.2f, r_pos_clust_max = %.2f;\n', ...
                    slack_ori_mil_pos,    slack_ori_mil_neg,    slack_ori_mic_pos,    loss_reg,        margin_cls_balance,        r_pos_tot*100,        max(r_pos_clust)*100 );
                disp(r_pos_clust);
end
count_iter_cccp = 0;
while true % ~flag_convg
    
    if VIS
        fprintf('\n CCCP_iter %i:', count_iter_cccp+1);
    end
    
    W0 = W;
    
    %% compute subgradient of constraints (A) and (C) for all positive bags
    uMat_mil_pos = compt_subg_mil_pos( W, indBagVec_pos, instDataMat_pos, list_indBag_pos, num_bag_pos, num_clust, dim_feat );
    uMat_mic_pos = compt_subg_mic_pos( W, indBagVec_pos, instDataMat_pos, list_indBag_pos, num_bag_pos, num_clust, dim_feat );
    
    %% solve the upper-bound function of <W> by cutting-plane algorithm;
    [W, loss_ub_dual, loss_ub_primal, slack_ub] = solve_ub_func_cuttingplane( W, uMat_mil_pos, uMat_mic_pos, uMat_cls_balance, kerMat_cls_balance, ...
                                                                               indBagVec_neg, instDataMat, list_indBag_neg, num_bag_neg, num_bag_pos, ...
                                                                                num_clust, dim_feat, M5IL, DEBUG );
                                               
%     if M5IL.compt_loss_ub
        all_loss_ub_full = [ all_loss_ub_full; loss_ub_primal ];
        all_loss_ub_end = [all_loss_ub_end; loss_ub_primal(end)];
%     end
    
    %% compute original loss;
        [loss_ori, slack_ori_mil_pos, slack_ori_mil_neg, slack_ori_mic_pos, loss_reg, margin_cls_balance, r_pos_clust, r_pos_tot] = ...
            compt_loss_ori( W, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat, list_indBag_neg, M5IL, true );
        disp(r_pos_clust);
        all_loss_ori = [all_loss_ori; loss_ori];
 
    %%
    if VIS
        fprintf(' loss_ub = %.2f, loss_ori = %.2f:\n', loss_ub_primal(end), all_loss_ori(end) );
        fprintf('  ub:  slack_mil_pos = %.2f, slack_mil_neg = %.2f, slack_mic_pos = %.2f;\n', slack_ub(1), slack_ub(2), slack_ub(3) );
%         fprintf(' ori:  slack_mil_pos = %.2f, slack_mil_neg = %.2f, slack_mic_pos = %.2f, loss_reg = %.2f, margin_cls_balance = %.2f;\n', slack_ori_mil_pos, slack_ori_mil_neg, slack_ori_mic_pos, loss_reg, margin_cls_balance );
        fprintf(' ori:  slack_mil_pos = %.2f, slack_mil_neg = %.2f, slack_mic_pos = %.2f, loss_reg = %.2f, margin_cls_balance = %.2f, r_pos_tot = %.2f, r_pos_clust_max = %.2f;\n', ...
                        slack_ori_mil_pos,    slack_ori_mil_neg,    slack_ori_mic_pos,    loss_reg,        margin_cls_balance,        r_pos_tot*100,        max(r_pos_clust)*100 );
    end
    
    count_iter_cccp = count_iter_cccp + 1;

    %% check convergence for stopping;
    if any( strcmp( 'detaLoss', M5IL.stop.criterion ) )
        deta_loss = abs(all_loss_ori(end) - all_loss_ori(end-1));
        if abs(deta_loss) <= M5IL.stop.minDetaLoss
            fprintf(' achieving minimum deta of loss;\n');
            break; % flag_convg = true;
        end                            
    end
    
    if any( strcmp( 'detaLossRate', M5IL.stop.criterion ) )
        deta_loss_rate = abs(all_loss_ori(end) - all_loss_ori(end-1)) / all_loss_ori(end-1);
        if abs(deta_loss_rate) <= M5IL.stop.minDetaLossRate
            fprintf(' achieving minimum deta of loss ratio;\n');
            break; % flag_convg = true;
        end                            
    end
    
    if any( strcmp( 'detaW', M5IL.stop.criterion ) )
        if sqrt( sum((W(:)-W0(:)).^2) ) <= M5IL.stop.minDetaW
            fprintf(' achieving minimum deta of L2-norm distance of model parameters;\n');
            break; % flag_convg = true;
        end                            
    end
    
    if any( strcmp( 'numIterCCCP', M5IL.stop.criterion ) )
        if count_iter_cccp >= M5IL.stop.maxNumIterCCCP
            fprintf(' achieving maximum number of CCCP iterations;\n');
            break; % flag_convg = true;
        end                            
    end        
    
end % end while;
fprintf('\n');

model.w = W;
model.train.alpha = M5IL.alpha;
model.train.beta = M5IL.beta;
model.train.gamma = M5IL.gamma;
model.train.zeta = M5IL.zeta;

% if M5IL.compt_loss_cccp
    info.loss_cccp = all_loss_ori;
% end
% if M5IL.compt_loss_ub
    info.loss_ub_full = all_loss_ub_full;
    info.loss_ub_end = all_loss_ub_end;
% end

info.train.num_iter_cccp = count_iter_cccp;
info.train.loss_cccp = all_loss_ori(end);
info.train.loss_ub = all_loss_ub_end(end);

info.train.slack_ori_mil_pos = slack_ori_mil_pos;
info.train.slack_ori_mil_neg = slack_ori_mil_neg;
info.train.slack_ori_mic_pos = slack_ori_mic_pos;
info.train.loss_reg = loss_reg;
info.train.margin_cls_balance = margin_cls_balance; 
info.train.r_pos_tot = r_pos_tot;
info.train.r_pos_clust = r_pos_clust;

end % end of function;

% %% get initial model ("specified");
% function [ wMat, loss ] = get_init_model_spec( ind_spec, all_wMat_Init, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat_neg, list_indBag_neg, M5IL )
% wMat = all_wMat_Init{ind_spec};
% loss = compt_loss_ori( wMat, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat_neg, list_indBag_neg, M5IL, false );
% end % end of function;

%% get initial model ("automatic");
function [ wMat, loss ] = get_init_model_auto( all_wMat_Init, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat_neg, list_indBag_neg, M5IL )
% loss = 0;
loss = inf;
wMat = [];
for i = 1: numel(all_wMat_Init)
    loss_i = compt_loss_ori( all_wMat_Init{i}, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat_neg, list_indBag_neg, M5IL, false );
%     if loss_i > loss
    if loss_i < loss
        wMat = all_wMat_Init{i};
        loss = loss_i;
    end
end % i;
end % end of function;

%% compute constraint on class balance for positive bags (constraint D);
function uVec = compt_const_balance_class( p, q, indBagVec_pos, instDataMat_pos, indValidVec_pos, list_indBag_pos, num_bag_pos, num_clust, dim_feat, class_data )
uVec = zeros([dim_feat num_clust],class_data);
for i = 1: num_bag_pos
     flag_inst_bag_i = (indBagVec_pos==list_indBag_pos(i));
     temp_u = mean( instDataMat_pos(flag_inst_bag_i & indValidVec_pos,:), 1 )';
     uVec(:,p) = uVec(:,p) + temp_u;
     uVec(:,q) = uVec(:,q) - temp_u;
end % i: index of positive bag;
uVec = uVec(:)/num_bag_pos;
end % end of function;

%% compute subgradient on MIL assumption for positive bags (constraint A);
function uMat_mil_pos = compt_subg_mil_pos( wMat, indBagVec_pos, instDataMat_pos, list_indBag_pos, num_bag_pos, num_clust, dim_feat )
[ignored_scoreVec_max, instDataMat_max, zVec_max] = find_max_score_mi( wMat, indBagVec_pos, instDataMat_pos, list_indBag_pos, num_bag_pos, dim_feat );
uMat_mil_pos = zeros([dim_feat*num_clust num_bag_pos], class(instDataMat_pos));
for i = 1: num_bag_pos
    uMat_mil_pos( dim_feat*(zVec_max(i)-1)+1: dim_feat*zVec_max(i), i ) = instDataMat_max(i,:)'; 
end
end % end of function;

%% compute subgradient on MIL assumption for positive bags (constraint C);
function uMat_mic_pos = compt_subg_mic_pos( wMat, indBagVec_pos, instDataMat_pos, list_indBag_pos, num_bag_pos, num_clust, dim_feat )
scoreMat_pos = instDataMat_pos*wMat;
indValidVec_pos = any(scoreMat_pos>0,2);

scoreMat_pos = bsxfun( @minus, scoreMat_pos, mean(scoreMat_pos,2) );

% uMat_mic_pos = zeros([dim_feat*num_clust num_bag_pos], class(instDataMat_pos));

S_cluster = (num_clust/(num_clust-1));

uMat_mic_pos = [];

count_const = 0;
for i = 1: num_bag_pos
    
    flag_inst_bag_i = (indBagVec_pos==list_indBag_pos(i));
    flag_inst_bag_i = flag_inst_bag_i&indValidVec_pos;
    
    indices_inst_bag_i = find(flag_inst_bag_i);
    
    if ~isempty(indices_inst_bag_i)
        count_const = count_const + 1;
        
        temp_scoreMat = scoreMat_pos(flag_inst_bag_i,:);
        [ignored_score_max, linearIndex_max] = max(temp_scoreMat(:));
        
        [ind_inst_max_bag_i, z_max] = ind2sub( size(temp_scoreMat), linearIndex_max );
        index_inst_max = indices_inst_bag_i( ind_inst_max_bag_i );
        instDataVec_max = instDataMat_pos(index_inst_max,:)';
        
        uMat_mic_pos = [uMat_mic_pos, zeros([dim_feat*num_clust 1], class(instDataMat_pos))];
        uMat_mic_pos( dim_feat*(z_max-1)+1: dim_feat*z_max, count_const ) = instDataVec_max;
        uMat_mic_pos( :, count_const ) = uMat_mic_pos( :, count_const ) - repmat(instDataVec_max,[num_clust 1])/num_clust;
        uMat_mic_pos( :, count_const ) = S_cluster*uMat_mic_pos( :, count_const );        
    end
    
end % i: index of bag;

end % end of function;

%% find the configuration (j,z_{ij}) with maximum score; 
function [scoreVec_max, instDataMat_max, zVec_max] = find_max_score_mi( wMat, indBagVec, instDataMat, list_indBag, num_bag, dim_feat )
scoreMat = instDataMat*wMat;

scoreVec_max = zeros([num_bag 1]);
instDataMat_max = zeros([num_bag dim_feat], class(instDataMat));
zVec_max = zeros([num_bag 1]);

% instIndVec_max = zeros([num_bag 1]);

for i = 1: num_bag
    
    flag_inst_bag_i= indBagVec==list_indBag(i);
    indices_inst_bag_i = find(flag_inst_bag_i);
    
    temp_scoreMat = scoreMat(flag_inst_bag_i,:);
    [scoreVec_max(i),linearIndex_max] = max(temp_scoreMat(:));
    
    [ind_inst_max_bag_i, zVec_max(i)] = ind2sub( size(temp_scoreMat), linearIndex_max );
    index_inst_max = indices_inst_bag_i( ind_inst_max_bag_i );
    instDataMat_max(i,:) = instDataMat(index_inst_max,:);
    
%     instIndVec_max(i) = index_inst_max;
    
end % i: index of bag;

end % end of fucntion;

%% solve upper-bound loss function by cutting-plane algorithm;
function [wMat, loss_dual, loss_primal, slack] = solve_ub_func_cuttingplane( wMat, uMat_mil_pos, uMat_mic_pos, uMat_ws, kerMat_ws, ...
                                                                              indBagVec_neg, instDataMat_neg, list_indBag_neg, num_bag_neg, num_bag_pos, ...
                                                                               num_clust, dim_feat, Params, DEBUG )
C = [Params.alpha; Params.beta; Params.gamma];
zeta = Params.zeta;
epsilon = Params.eps;
siz_incremt_ws = Params.siz_incremt_ws;

slack = zeros([3 1]);
loss_dual = [];
loss_primal = [];
                   
wVec = wMat(:)';
siz_ws_tot = size(uMat_ws,2);
if size(kerMat_ws,1)~=siz_ws_tot || size(kerMat_ws,2)~=siz_ws_tot
    error('Error: the size of kernel matrix does not match;\n');
end
class_data = class(uMat_ws);
bVec_ws = -zeta * ones( [siz_ws_tot 1], class_data );
indices_const_ws = cell([3 1]);

count_const_ws = siz_ws_tot;
if count_const_ws == siz_ws_tot
    [kerMat_ws, siz_ws_tot]  = incremt_ker_mat( kerMat_ws, siz_incremt_ws );
end

if isempty(uMat_mic_pos)
    num_const_term = 2;
else
    num_const_term = 3;
end

flag_fmc = false;
for s = 1: num_const_term
    % find the most violated constraint;
    switch s
        case 1
            % (1) for constraint A:
            value_vlt_new = find_most_vlt_const_pos( wVec, uMat_mil_pos, num_bag_pos );
                
        case 2
            % (2) for constraint B:
            value_vlt_new = find_most_vlt_const_mil_neg( wMat, wVec, indBagVec_neg, instDataMat_neg, list_indBag_neg, num_bag_neg, dim_feat, num_clust, class_data );
                
        case 3
            % (3) for constraint C:
            value_vlt_new = find_most_vlt_const_pos( wVec, uMat_mic_pos, num_bag_pos );
                
        otherwise
                error('Error: unknown constraint term;\n');
    end
    if value_vlt_new > (slack(s) + epsilon)
        flag_fmc = ~flag_fmc;
        break;
    end
end

if ~flag_fmc
    [lamVec, loss_dual_iter] = solve_qp_dual_v1( kerMat_ws(1:count_const_ws,1:count_const_ws), bVec_ws, indices_const_ws, C, Params.solveQP );
    
    disp(lamVec');
    
    loss_dual = [loss_dual; loss_dual_iter];
    % update <W> and slack variables;
    wVec = uMat_ws*lamVec;
    wMat = reshape(wVec,[dim_feat num_clust]);
    wVec = wVec';
    for j = 1: length(slack)
        if ~isempty(indices_const_ws{j})
            slack(j) = max( max( bVec_ws(indices_const_ws{j})' - wVec*uMat_ws(:,indices_const_ws{j}) ), 0 );
        end
    end    
    % compute primal loss;
    loss_primal_iter = (wVec*wVec')/2;
    for j = 1: length(slack)
        loss_primal_iter = loss_primal_iter + C(j)*slack(j);
    end
    loss_primal = [loss_primal; loss_primal_iter];
    if abs(loss_primal_iter-loss_dual_iter)>epsilon
        if DEBUG
            fprintf('Warning: the difference loss (%f) of primal and dual is larger than epsilon (%f);\n', ...
                     abs(loss_primal_iter-loss_dual_iter), epsilon);
%             pause;            
        else
%             error('Error: the losses of primal and dual problems do not match;\n');
        end
    end    
else
    lamVec = [];
end   

flag_go_on = true;
count_iter = 0;
while flag_go_on
    
    fprintf(' .%i', count_iter+1);
    
    flag_go_on = false;    
    
    for s = 1: num_const_term
        
        % find the most violated constraint;
        switch s
            case 1
                % (1) for constraint A:
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_pos( wVec, uMat_mil_pos, num_bag_pos );
                
            case 2
                % (2) for constraint B:
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_mil_neg( wMat, wVec, indBagVec_neg, instDataMat_neg, list_indBag_neg, num_bag_neg, dim_feat, num_clust, class_data );
                
            case 3
                % (3) for constraint C:
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_pos( wVec, uMat_mic_pos, num_bag_pos );
                
            otherwise
                error('Error: unknown constraint term;\n');
        end
        
        if value_vlt_new > (slack(s) + epsilon)
            % A new cutting plane is found, and add it to working set;
            count_const_ws = count_const_ws + 1;              
            uMat_ws = [uMat_ws, uVec_new];
            kerVec_new = uVec_new'*uMat_ws;
            kerMat_ws(count_const_ws,1:count_const_ws) = kerVec_new;
            kerMat_ws(1:count_const_ws,count_const_ws) = kerVec_new';
            bVec_ws = [bVec_ws; b_new];            
            indices_const_ws{s} = [indices_const_ws{s}; count_const_ws];
        
            % solve QP in dual on working set;
            if ~isempty(lamVec) && Params.solveQP.init_from_last
                lamVec_init = [lamVec; 0];
            else
                lamVec_init = [];
            end
            [lamVec, loss_dual_iter] = solve_qp_dual_v1( kerMat_ws(1:count_const_ws,1:count_const_ws), bVec_ws, indices_const_ws, C, Params.solveQP, lamVec_init );
            
%             disp(lamVec(1:9)')
            
            loss_dual = [loss_dual; loss_dual_iter];
            
            % update <W> and slack variables;
            wVec = uMat_ws*lamVec;
            wMat = reshape(wVec,[dim_feat num_clust]);
            wVec = wVec';
            for j = 1: length(slack)
                if ~isempty(indices_const_ws{j})
                    slack(j) = max( max( bVec_ws(indices_const_ws{j})' - wVec*uMat_ws(:,indices_const_ws{j}) ), 0 );
                end
            end
            
            % compute primal loss;
            loss_primal_iter = (wVec*wVec')/2;
            for j = 1: length(slack)
                loss_primal_iter = loss_primal_iter + C(j)*slack(j);
            end
            loss_primal = [loss_primal; loss_primal_iter];
            if abs(loss_primal_iter-loss_dual_iter)>epsilon
                if DEBUG
                    fprintf('Warning: the difference loss (%f) of primal and dual is larger than epsilon (%f);\n', ...
                     abs(loss_primal_iter-loss_dual_iter), epsilon);
%                     pause;
                else
%                     error('Error: the losses of primal and dual problems do not match;\n');
                end
            end
        
            if count_const_ws >= siz_ws_tot
                [kerMat_ws, siz_ws_tot]  = incremt_ker_mat( kerMat_ws, siz_incremt_ws );
            end
        
            flag_go_on = flag_go_on | true;
        end
        
    end
    
    count_iter = count_iter + 1;
       
end

end % end of function;

%% find the most violated constraint for (A) or (C);
function [value_vlt, uVec, b] = find_most_vlt_const_pos( wVec, uMat, N )
indVec_const_violated = ( wVec*uMat < 1 );
uVec = sum( uMat(:,indVec_const_violated), 2 )/N;
b = sum(indVec_const_violated)/N; 
value_vlt = b - wVec*uVec;
end % end of function;

%% find the most violated constraint for (B);
function [value_vlt, uVec, b] = find_most_vlt_const_mil_neg( wMat, wVec, indBagVec_neg, instDataMat_neg, list_indBag_neg, num_bag_neg, dim_feat, num_clust, class_data )
[scoreVec_max_bag_neg, instDataMat_max_bag_neg, zVec_max_bag_neg] = find_max_score_mi( wMat, indBagVec_neg, instDataMat_neg, list_indBag_neg, num_bag_neg, dim_feat );
indVec_bag_violated = ( scoreVec_max_bag_neg > -1 );
uVec = zeros( [dim_feat*num_clust 1], class_data );
for i = 1: num_bag_neg
    if indVec_bag_violated(i)
        uVec( dim_feat*(zVec_max_bag_neg(i)-1)+1: dim_feat*zVec_max_bag_neg(i) ) = ...
        uVec( dim_feat*(zVec_max_bag_neg(i)-1)+1: dim_feat*zVec_max_bag_neg(i) ) + instDataMat_max_bag_neg(i,:)'; 
    end
end
uVec = -uVec/num_bag_neg;
b = sum(indVec_bag_violated)/num_bag_neg;
value_vlt = b - wVec*uVec;
end % end of function;

%% increase the size of kernel matrix;
function [kerMat, siz]  = incremt_ker_mat( kerMat, siz_incremt )
siz = size(kerMat,1);
copy_kerMat = kerMat;
kerMat = zeros( siz+siz_incremt, class(copy_kerMat) );
kerMat(1:siz,1:siz) = copy_kerMat;
siz = siz+siz_incremt;
clear('copy_kerMat');
end % end of function;

%% compute loss of original problem in CCCP iteration;
function [loss_tot, slack_mil_pos, slack_mil_neg, slack_mic_pos, loss_reg, margin_cls_balance, r_pos_clust, r_pos_tot] = compt_loss_ori( wMat, indBagVec_pos, instDataMat_pos, list_indBag_pos, indBagVec_neg, instDataMat_neg, list_indBag_neg, Params, check_cls_balance )
if ~exist('check_cls_balance', 'var')
    check_cls_balance = true;
end

num_clust = size(wMat,2);
num_bag_pos = length(list_indBag_pos);
num_bag_neg = length(list_indBag_neg);

alpha = Params.alpha;
beta = Params.beta;
gamma = Params.gamma;
zeta = Params.zeta;

%%
scoreMat_mi_pos = instDataMat_pos*wMat;

% scoreMat_mi_pos(1:10,:)


%% check class balance constraint;
meanScoreVec_cls_pos = zeros([1 num_clust]);
for i = 1: num_bag_pos
    temp_scoreMat_bag_i = scoreMat_mi_pos( (indBagVec_pos==list_indBag_pos(i)), : );
    meanScoreVec_cls_pos = meanScoreVec_cls_pos + mean(temp_scoreMat_bag_i, 1);    
end
meanScoreVec_cls_pos = meanScoreVec_cls_pos / num_bag_pos;

disp(meanScoreVec_cls_pos);

margin_cls_balance = 0.;
for p = 1: num_clust
    for q = 1: num_clust
        diff_mean_score_p_q_pos = abs( meanScoreVec_cls_pos(p) - meanScoreVec_cls_pos(q) );
        if diff_mean_score_p_q_pos > zeta
            if check_cls_balance
%                 error('Error: the constraint of class balance is violated;\n');
                fprintf('Error: the constraint of class balance is violated;\n');
            end                
        end
        margin_cls_balance = margin_cls_balance + diff_mean_score_p_q_pos;
    end
end
margin_cls_balance = margin_cls_balance / (num_clust^2);

% get the ratio of each modality;
flagVec_inst_pos = any( scoreMat_mi_pos>0, 2 );
r_pos_tot = sum(flagVec_inst_pos)/length(flagVec_inst_pos);

scoreMat_inst_pos = scoreMat_mi_pos(flagVec_inst_pos,:);
[ignored_scoreVec_max, indVec_clust_inst_pos] = max( scoreMat_inst_pos, [], 2 );
r_pos_clust = zeros([1 num_clust]);
for p = 1: num_clust
    r_pos_clust(p) = sum(indVec_clust_inst_pos==p);
end % p: index of cluster;
r_pos_clust = r_pos_clust / sum(r_pos_clust);

%%
loss_tot = 0.;

%% for regularization term;
wVec = wMat(:)';
loss_reg = (wVec*wVec') / 2;
loss_tot = loss_tot + loss_reg;

%% for constraint (A);
slack_mil_pos = 0.;
for i = 1: num_bag_pos
      score_max_mil_pos_bag_i = max( max( scoreMat_mi_pos( (indBagVec_pos==list_indBag_pos(i)), : ) ) ); 
      slack_mil_pos = slack_mil_pos + max( (1-score_max_mil_pos_bag_i), 0 );
end % i: index of bag;
slack_mil_pos = slack_mil_pos / num_bag_pos;
loss_tot = loss_tot + alpha * slack_mil_pos;

%% for constraint (B);
scoreMat_mi_neg = instDataMat_neg*wMat;
slack_mil_neg = 0.;
for i = 1: num_bag_neg
      score_max_mil_neg_bag_i = max( max( scoreMat_mi_neg( (indBagVec_neg==list_indBag_neg(i)), : ) ) );
      slack_mil_neg = slack_mil_neg + max( (1+score_max_mil_neg_bag_i), 0 );
end % i: index of bag;
slack_mil_neg = slack_mil_neg / num_bag_neg;
loss_tot = loss_tot + beta * slack_mil_neg;

%% for constraint (C);
tempScoreVec_max_mic_pos = max(scoreMat_mi_pos, [], 2) - mean(scoreMat_mi_pos,2);
S_cluster = (num_clust/(num_clust-1));
slack_mic_pos = 0.;
for i = 1: num_bag_pos
    indVec_pos_inst = any( scoreMat_mi_pos(indBagVec_pos==list_indBag_pos(i),:)>0, 2 );
    if any(indVec_pos_inst)
        tempScoreVec_max_pos_bag_i = tempScoreVec_max_mic_pos( indBagVec_pos==list_indBag_pos(i) );
        score_max_mic_pos_bag_i = S_cluster * max( tempScoreVec_max_pos_bag_i(indVec_pos_inst) );   
        slack_mic_pos = slack_mic_pos + max( (1-score_max_mic_pos_bag_i), 0 );
    end
end % i: index of bag;
slack_mic_pos = slack_mic_pos / num_bag_pos;
loss_tot = loss_tot + gamma * slack_mic_pos;

end % end of function;