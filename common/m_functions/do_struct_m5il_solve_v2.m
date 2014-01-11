% wyw 09/20/2013 @MSRA
% solve the problem of multi-model graph structured multiple instance model learning
% low mem version

% input:
% X: [instance_num, instance_feat_dim] all the instance in all the bags
% instance_bag_labels: [instance_num,1] the bag label of the instances
% bag_labels_binary: [bag_num,1], the bag pos and neg labels of current training data
% config: all the params and initial data

% M5IL.init.type = 'spec'; %'auto';
% M5IL.init.ind_spec = 1;
% M5IL.init.wInit = [];

% M5IL.stop.criterion = {'numIterCCCP', 'detaLoss', 'detaLossRate'};
% M5IL.stop.maxNumIterCCCP = 10;
% M5IL.stop.minDetaLoss = .1;
% M5IL.stop.minDetaLossRate = .01;

% M5IL.alpha = 2^10;
% M5IL.beta = 2^12;
% M5IL.gamma = 2^5;
% M5IL.zeta = 5;

% M5IL.eps = .02;
% M5IL.siz_incremt_ws = 100;

% M5IL.solveQP.init_from_last = true;      %
% M5IL.solveQP.solver = 'matlab';          % 'matlab', 'svm-struct';

% M5IL.compt_loss_cccp = true;
% M5IL.compt_loss_ub = true;

% output: the model W: [psi_dim,config.graph.node_num]

% specify the problem
% constrait:
% A) 1: negative bags all < -1
% B) 2: max of positive bags all > 1
% C) 3: the difference of channels on pos bags
% D) 4: avoid plant solve

% version info 
% v0 -> v1: v0 low mem, load the bag each time form disk, v1,high mem:
% load all the bag in memory
% v2 graph -> only node 
function [model, info] = do_struct_m5il_solve_v2(bag_datas_pos, bag_datas_neg, W_init, config)
%%%%%%%%%%%%%%%%%%%%%%%%%%
% all params list
% config.using_parallel
% config.method:  for infer 'traversal' or 'TRW-S'
% config.channel_num
% config.graph.node_num
% config.graph.node_dim
% config.graph.edge_num = config.graph.node_num*(config.graph.node_num-1)/2
% config.graph.edge_dim
% config.graph.psi_dim = config.graph.node_num*config.graph.node_dim + config.graph.edge_num*config.graph.edge_dim;
% config.C_neg = 2^12; constrait A
% config.C_pos = 2^10; constrait B
% config.C_div = 2^5; constraint C
% config.zeta = 5; constrait D
% config.stop.criterion = {'numIterCCCP', 'detaLoss', 'detaLossRate'};
% config.stop.maxNumIterCCCP = 10;
% config.stop.minDetaLoss = .1;
% config.stop.minDetaLossRate = .01;
% config.eps = .02;
% config.siz_incremt_ws = 100;
% config.solveQP.init_from_last = true;      %
% config.solveQP.solver = 'matlab';          % 'matlab', 'svm-struct';
%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%
% prepare varibles
%%%%%%%%%%%%%%%%%%%
config.graph.psi_dim = config.graph.node_num*config.graph.node_dim + config.graph.edge_num*config.graph.edge_dim;
%%%%%%%%%%%%%%%%%
% CCCP initial
%%%%%%%%%%%%%%%%

% init the kernel part of constrait D, need to travelling the hidden space of positive bags
[kernel_constrait_D,u_D,sum_mean_psi_pos_bags] = compute_kernel_constrait_D(bag_datas_pos,config);

% init W_0 compute the initial energy, select the smallest initial one

energy_CCCP_rec = [];
energy_cutting_plane_all_rec = {};
energy_cutting_plane_end_rec = [];


energy = Inf;
for i_init = 1:length(W_init)
    
    fprintf('initializing %d ...\n',i_init);
    
    [energy_i,slack_neg_i,slack_pos_i,slack_div_i,balance_score_i,TPR_i,TNR_i,channel_distribute_i,subgradient_constrait_B_i,subgradient_constrait_C_i] =...
        compute_energy(bag_datas_pos,bag_datas_neg,W_init{i_init},sum_mean_psi_pos_bags,config);
    
    if energy_i < energy
        W = W_init{i_init};
        energy = energy_i;
        slack_neg = slack_neg_i;
        slack_pos = slack_pos_i;
        slack_div = slack_div_i;
        balance_score = balance_score_i;
        TPR = TPR_i;
        TNR = TNR_i;
        channel_distribute = channel_distribute_i;
        subgradient_constrait_B = subgradient_constrait_B_i;
        subgradient_constrait_C = subgradient_constrait_C_i;
    end
end
    

energy_CCCP_rec = [energy_CCCP_rec;energy];

fprintf('CCCP iter 0:  energy:%f slack_neg = %.2f, slack_pos = %.2f, slack_div = %.2f, \n           balance_score = %.2f, TPR = %.2f, TNR = %.2f, channel_distribute =', ...
   energy, slack_neg, slack_pos,    slack_div,    balance_score, TPR, TNR); disp(channel_distribute);

%%%%%%%%%%%%%%%%%
% CCCP main loop
%%%%%%%%%%%%%%%%%
iter_CCCP = 0;
while true
    iter_CCCP = iter_CCCP +1;
    fprintf('CCCP iter %2d: computing cutting_plane ',iter_CCCP);
    W_0 = W;
    
    %--------------------
    % cutting plane
    %--------------------
    [W, energy_cutting_plane_dual, energy_cutting_plane_primal, slack_neg,slack_pos,slack_div] = ...
        solve_cuttingplane( W, subgradient_constrait_B,subgradient_constrait_C, kernel_constrait_D, u_D, bag_datas_pos, bag_datas_neg, config );
    
    energy_cutting_plane_all_rec = [ energy_cutting_plane_all_rec; energy_cutting_plane_primal ];
    energy_cutting_plane_end_rec = [energy_cutting_plane_end_rec; energy_cutting_plane_primal(end)];
    % fprintf(' energy cutting plane= %.2f, energy CCCP = %.2f:\n', energy_cutting_plane_primal(end), energy_CCCP_rec(end) );
    % fprintf(' slack_neg = %.2f, slack_pos = %.2f, slack_div = %.2f;\n', slack_neg,slack_pos,slack_div );
    
    
    %---------------------------------------------------
    % computing subgradient for constrait B and C
    %---------------------------------------------------
    [energy_CCCP,slack_neg,slack_pos,slack_div,balance_score,TPR,TNR,channel_distribute,subgradient_constrait_B,subgradient_constrait_C] = ...
        compute_energy(bag_datas_pos,bag_datas_neg,W,sum_mean_psi_pos_bags,config);
    energy_CCCP_rec = [energy_CCCP_rec; energy_CCCP];
    
    fprintf('\nCCCP iter %2d: energy:%f slack_neg = %.2f, slack_pos = %.2f, slack_div = %.2f, \n             balance_score = %.2f, TPR = %.2f, TNR = %.2f, channel_distribute =', ...
    iter_CCCP, energy_CCCP, slack_neg, slack_pos, slack_div, balance_score, TPR, TNR); disp(channel_distribute);
    
       
    %--------------------------
    % check whether to stop
    %--------------------------
    if any( strcmp( 'detaLoss', config.stop.criterion ) )
        deta_energy = abs(energy_CCCP_rec(end) - energy_CCCP_rec(end-1));
        if abs(deta_energy) <= config.stop.minDetaLoss
            fprintf(' achieving minimum deta of loss;\n');
            break; % flag_convg = true;
        end
    end
    
    if any( strcmp( 'detaLossRate', config.stop.criterion ) )
        deta_energy_rate = abs(energy_CCCP_rec(end) - energy_CCCP_rec(end-1)) / energy_CCCP_rec(end-1);
        if abs(deta_energy_rate) <= config.stop.minDetaLossRate
            fprintf(' achieving minimum deta of loss ratio;\n');
            break; % flag_convg = true;
        end
    end
    
    if any( strcmp( 'detaW', config.stop.criterion ) )
        if sqrt( sum((W(:)-W_0(:)).^2) ) <= config.stop.minDetaW
            fprintf(' achieving minimum deta of L2-norm distance of model parameters;\n');
            break; % flag_convg = true;
        end
    end
    
    if any( strcmp( 'numIterCCCP', config.stop.criterion ) )
        if iter_CCCP >= config.stop.maxNumIterCCCP
            fprintf(' achieving maximum number of CCCP iterations;\n');
            break; % flag_convg = true;
        end
    end
    
end % main loop of CCCP

model.W = W;
info.energy = energy_CCCP_rec;

end % function end: do_struct_m5il_solve

%%
function [W_new, energy_cutting_plane_dual, energy_cutting_plane_primal, slack_neg,slack_pos,slack_div] = solve_cuttingplane( W, subgradient_constrait_B,subgradient_constrait_C, kerMat_ws, uMat_ws, bag_datas_pos,bag_datas_neg, config )

DEBUG = false;

bag_pos_num = length(bag_datas_pos);

C = [config.C_neg;config.C_pos;config.C_div];
zeta = config.zeta;
epsilon = config.eps;
siz_incremt_ws = config.siz_incremt_ws;

slack = zeros(3,1);

energy_cutting_plane_dual = [];
energy_cutting_plane_primal = [];

W_vec = W(:);
siz_ws_tot = size(uMat_ws,1);
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

if isempty(subgradient_constrait_C)
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
            W_new = reshape(W_vec,size(W));
            value_vlt_new = find_most_vlt_const_neg( W_new, bag_datas_neg, config);            
        case 2
            % (2) for constraint B:
            value_vlt_new = find_most_vlt_const_pos( W_vec, subgradient_constrait_B, bag_pos_num );               
        case 3
            % (3) for constraint C:
            value_vlt_new = find_most_vlt_const_pos( W_vec, subgradient_constrait_C, bag_pos_num );
                
        otherwise
                error('Error: unknown constraint term;\n');
    end
    if value_vlt_new > (slack(s) + epsilon)
        flag_fmc = ~flag_fmc;
        break;
    end
end

if ~flag_fmc
    [lamVec, energy_dual_iter] = solve_qp_dual_v1( kerMat_ws(1:count_const_ws,1:count_const_ws), bVec_ws, indices_const_ws, C, config.solveQP );
    energy_cutting_plane_dual = [energy_cutting_plane_dual; energy_dual_iter];
    % update <W> and slack variables;
    W_vec = uMat_ws'*lamVec;
   
    for j = 1: length(slack)
        if ~isempty(indices_const_ws{j})
            slack(j) = max( max( bVec_ws(indices_const_ws{j})' - (uMat_ws(indices_const_ws{j},:)*W_vec)' ), 0 );
        end
    end    
    % compute primal loss;
    energy_primal_iter = (W_vec'*W_vec)/2;
    for j = 1: length(slack)
        energy_primal_iter = energy_primal_iter + C(j)*slack(j);
    end
    energy_cutting_plane_primal = [energy_cutting_plane_primal; energy_primal_iter];
    if abs(energy_primal_iter-energy_dual_iter)>epsilon
        if DEBUG
            fprintf('Warning: the difference loss (%f) of primal and dual is larger than epsilon (%f);\n', ...
                     abs(energy_primal_iter-energy_dual_iter), epsilon);
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
                W_new = reshape(W_vec,size(W));
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_neg( W_new, bag_datas_neg, config);                      
            case 2
                % (2) for constraint B:              
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_pos( W_vec,subgradient_constrait_B, bag_pos_num );
            case 3
                % (3) for constraint C:
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_pos( W_vec, subgradient_constrait_C, bag_pos_num );
                
            otherwise
                error('Error: unknown constraint term;\n');
        end
%         fprintf('constrait:%d vlt:%f\n',s,value_vlt_new);
        if value_vlt_new > (slack(s) + epsilon)
            % A new cutting plane is found, and add it to working set;
            count_const_ws = count_const_ws + 1;              
            uMat_ws = [uMat_ws; uVec_new];
            kerVec_new = uVec_new*uMat_ws';
            kerMat_ws(count_const_ws,1:count_const_ws) = kerVec_new;
            kerMat_ws(1:count_const_ws,count_const_ws) = kerVec_new';
            bVec_ws = [bVec_ws; b_new];            
            indices_const_ws{s} = [indices_const_ws{s}; count_const_ws];
        
            % solve QP in dual on working set;
            if ~isempty(lamVec) && config.solveQP.init_from_last
                lamVec_init = [lamVec; 0];
            else
                lamVec_init = [];
            end
            [lamVec, energy_dual_iter] = solve_qp_dual_v1( kerMat_ws(1:count_const_ws,1:count_const_ws), bVec_ws, indices_const_ws, C, config.solveQP, lamVec_init );
            energy_cutting_plane_dual = [energy_cutting_plane_dual; energy_dual_iter];
            
            % update <W> and slack variables;
            W_vec = uMat_ws'*lamVec;
            
            for j = 1: length(slack)
                if ~isempty(indices_const_ws{j})
                    slack(j) = max( max( bVec_ws(indices_const_ws{j})' - (uMat_ws(indices_const_ws{j},:)*W_vec)' ), 0 );
                end
            end
            
            % compute primal loss;
            energy_primal_iter = (W_vec'*W_vec)/2;
            for j = 1: length(slack)
                energy_primal_iter = energy_primal_iter + C(j)*slack(j);
            end
            energy_cutting_plane_primal = [energy_cutting_plane_primal; energy_primal_iter];
            if abs(energy_primal_iter-energy_dual_iter)>epsilon
                if DEBUG
                    fprintf('Warning: the difference loss (%f) of primal and dual is larger than epsilon (%f);\n', ...
                     abs(energy_primal_iter-energy_dual_iter), epsilon);
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

slack_neg = slack(1);
slack_pos = slack(2);
slack_div = slack(3);

W_new = reshape(W_vec,size(W));

end

%% find the most violated constraint for (B) or (C);
function [value_vlt, uVec, b] = find_most_vlt_const_pos( W_vec, uMat, N )
indVec_const_violated = ( uMat*W_vec < 1 );
uVec = sum( uMat(indVec_const_violated,:),1 )/N;
b = sum(indVec_const_violated)/N; 
value_vlt = b - uVec*W_vec;
end % end of function;

%% find the most violated constraint for (A);
function [value_vlt, uVec, b] = find_most_vlt_const_neg( W, bag_datas_neg, config)      

bags_num_neg = length(bag_datas_neg);
scores_neg_all_channel = zeros(bags_num_neg,config.channel_num);
psis = cell(bags_num_neg,1);

if config.using_parallel
     parfor i = 1:bags_num_neg
        [scores_neg_all_channel(i,:),psis{i}] = infer_bag_score(bag_datas_neg{i},W,config);
    end
else
    for i = 1:bags_num_neg
        [scores_neg_all_channel(i,:),psis{i}] = infer_bag_score(bag_datas_neg{i},W,config);
    end
end
scores_neg = max(scores_neg_all_channel,[],2);
indVec_bag_violated = ( scores_neg > -1 );

uVec = sum(cell2mat(psis(indVec_bag_violated)),1);
uVec = -uVec/bags_num_neg;

b = sum(indVec_bag_violated)/bags_num_neg;
value_vlt = b - uVec*W(:);
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

%% computing model energy
function [energy_CCCP,slack_neg,slack_pos,slack_div,balance_score,TPR,TNR,channel_distribute,subgradient_constrait_B,subgradient_constrait_C] = compute_energy(bag_datas_pos,bag_datas_neg,W,sum_mean_psi_pos_bags,config)

% fprintf('computing energy... ');
energy_CCCP = 0;
% ------------------------------------------------------------
% regular part
% ------------------------------------------------------------
energy_W = sum(sum(W.*W) )/2;
energy_CCCP = energy_CCCP + energy_W;

% ------------------------------------------------------------
% constrait D 
% ------------------------------------------------------------
% fprintf('balance constrait... ');
mean_pos_bag_scores = sum_mean_psi_pos_bags*W;
balance_score = 0;
for p = 1:config.channel_num
    for q = p+1:config.channel_num
        balance_score = balance_score + abs(mean_pos_bag_scores(p)-mean_pos_bag_scores(q));
        if abs(mean_pos_bag_scores(p)-mean_pos_bag_scores(q)) > config.zeta;
            fprintf('mean_pos_bag_scores: ');
            disp(mean_pos_bag_scores);
            fprintf('WRONG!!! balance constrait not satisfied! \n');           
        end     
    end
end

% ------------------------------------------------------------
% constrait A , all negtive bags
% ------------------------------------------------------------
% fprintf('neg constrait... ');
bags_num_neg = length(bag_datas_neg);
scores_neg_all_channel = zeros(bags_num_neg,config.channel_num);

if config.using_parallel
     parfor i = 1:bags_num_neg
        scores_neg_all_channel(i,:) = infer_bag_score(bag_datas_neg{i},W,config);
    end
else
    for i = 1:bags_num_neg
        scores_neg_all_channel(i,:) = infer_bag_score(bag_datas_neg{i},W,config);
    end
end
scores_neg = max(scores_neg_all_channel,[],2);

slacks_neg = max(1+scores_neg,0);
slack_neg = mean(slacks_neg);
energy_CCCP = energy_CCCP + config.C_neg*slack_neg;


% ------------------------------------------------------------
% constrait B and C 
% ------------------------------------------------------------
% fprintf('pos constrait... ');


bags_num_pos = length(bag_datas_pos);
scores_pos_all_channel = zeros(bags_num_pos,config.channel_num);
divs = zeros(bags_num_pos,1);
psis_max_div = cell(bags_num_pos,1);
psis_max_score = cell(bags_num_pos,1);

if config.using_parallel
     parfor i = 1:bags_num_pos
        [scores_pos_all_channel(i,:),psis_max_score{i},divs(i),psis_max_div{i}]= infer_bag_score_div(bag_datas_pos{i},W,config);
    end
else
    for i = 1:bags_num_pos
        [scores_pos_all_channel(i,:),psis_max_score{i},divs(i),psis_max_div{i}]= infer_bag_score_div(bag_datas_pos{i},W,config);
    end
end

[scores_pos,channel_max_scores ]= max(scores_pos_all_channel,[],2);

slacks_pos = max(1-scores_pos,0);
slack_pos = mean(slacks_pos);
energy_CCCP = energy_CCCP + config.C_pos*slack_pos;

slacks_div = max(1-divs,0);
slack_div = mean(slacks_div);
energy_CCCP = energy_CCCP + config.C_div*slack_div;

subgradient_constrait_B = cell2mat(psis_max_score);
subgradient_constrait_C = cell2mat(psis_max_div);

% ------------------------------------------------------------
% classification info
% ------------------------------------------------------------

TPR = mean(slacks_pos == 0);
TNR = mean(slacks_neg == 0);


channel_distribute = zeros(1,config.channel_num);
for i_channel = 1:config.channel_num
channel_distribute(i_channel) = mean(channel_max_scores == i_channel);
end

end



%% for constrait B and C, infer max score and max div of a positive bag

function [scores,psi_max_score, div, psi_max_div] = infer_bag_score_div(bag_data,W,config)

parts_features = bag_data{1};
parts_pair_features = bag_data{2};

parts_features = [parts_features,ones(size(parts_features,1),1)];

if size(parts_features,2)  ~= config.graph.node_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
    return;
end


if size(parts_pair_features,2) ~= config.graph.edge_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
    return;
end

switch config.method
    case 'TRW-S'
        
    case 'traversal'
        
        phi_nodes_all = {config.channel_num,1};
        phi_edges_all = {config.channel_num,1};
        for i_channel = 1:config.channel_num
            
            % W_nodes [node_dim,node_num]
            % W_edges [edge_dim,edge_num]
            [W_nodes,W_edges] = get_graph_W (W(:,i_channel),config);
            
            phi_nodes_all{i_channel} = parts_features*W_nodes;
            phi_edges_all{i_channel} = parts_pair_features*W_edges;
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         channel_num = config.channel_num;
%         save('debug_MRF_infer_traversal.mat','phi_nodes_all','phi_edges_all','channel_num');
        [scores,node_ass_idx_max_score,div,channel_max_div,node_ass_idx_max_div] =...
            MRF_infer_div_traversal_mex(phi_nodes_all,phi_edges_all,config.channel_num);
        
    otherwise
        fprintf('WRONG!!! the infer method not exist!\n');
        return;
end

[~,channel_max_score] = max(scores);

psi_max_score = zeros(1,config.graph.psi_dim*config.channel_num);
psi = [];
for i_node = 1:config.graph.node_num
    psi = [psi,parts_features(node_ass_idx_max_score(i_node),:)];
end
for p = 1:config.graph.node_num
    parts_id_p = node_ass_idx_max_score(p);
    for q = p+1:config.graph.node_num
        parts_id_q = node_ass_idx_max_score(q);
        
        min_id = min(parts_id_p,parts_id_q);
        max_id = max(parts_id_p,parts_id_q);
        
        psi = [psi, parts_pair_features(get_pair_ind(min_id,max_id,size(parts_features,1)),:)];
    end
end
psi_max_score( (channel_max_score-1)*config.graph.psi_dim+1:channel_max_score*config.graph.psi_dim )  = psi;


% if the div infer is valid
if sum(node_ass_idx_max_div == 0)
    % no pos ass
    psi_max_div = [];
    
else
    
    psi_max_div = zeros(1,config.graph.psi_dim * config.channel_num);
    psi = [];
    for i_node = 1:config.graph.node_num
        psi = [psi,parts_features(node_ass_idx_max_div(i_node),:)];
    end
    for p = 1:config.graph.node_num
        parts_id_p = node_ass_idx_max_div(p);
        for q = p+1:config.graph.node_num
            parts_id_q = node_ass_idx_max_div(q);
            
            min_id = min(parts_id_p,parts_id_q);
            max_id = max(parts_id_p,parts_id_q);
            
            psi = [psi, parts_pair_features(get_pair_ind(min_id,max_id,size(parts_features,1)),:)];
        end
    end
    psi_max_div( (channel_max_div-1)*config.graph.psi_dim+1:channel_max_div*config.graph.psi_dim )  = psi;
    
    psi_max_div_all = zeros(1,config.graph.psi_dim*config.channel_num);
    for i_channel = 1:config.channel_num
        psi_max_div_all( (i_channel-1)*config.graph.psi_dim+1:i_channel*config.graph.psi_dim )  = psi;       
    end
    
    psi_max_div = (psi_max_div - psi_max_div_all/config.channel_num)*(config.channel_num/(config.channel_num - 1));
end


end % function end


%% for constrait A, infer max score of a negative bag
% output: 
% scores [1,channel_num]
% psi_max_score, the related psi of the max score channel
function [scores,psi_max_score] = infer_bag_score(bag_data,W,config)

parts_features = bag_data{1};
parts_pair_features = bag_data{2};


parts_features = [parts_features,ones(size(parts_features,1),1)];

if size(parts_features,2)  ~= config.graph.node_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
    return;
end


if size(parts_pair_features,2) ~= config.graph.edge_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
    return;
end

scores = zeros(1,config.channel_num);

switch config.method
    case 'TRW-S'
        
        phi_nodes_all = {config.channel_num,1};
        phi_edges_all = {config.channel_num,1};
        for i_channel = 1:config.channel_num
            
            % W_nodes [node_dim,node_num]
            % W_edges [edge_dim,edge_num]
            [W_nodes,W_edges] = get_graph_W (W(:,i_channel),config);
            
            % TRW-S find the min energy
            phi_nodes_all{i_channel} = -parts_features*W_nodes;
            phi_edges_all{i_channel} = -parts_pair_features*W_edges;
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [scores,node_ass_idx] = MRF_infer_TRW_S_mex(phi_nodes_all,phi_edges_all,config.channel_num);
        
    case 'traversal'
        
        phi_nodes_all = {config.channel_num,1};
        phi_edges_all = {config.channel_num,1};
        for i_channel = 1:config.channel_num
            
            % W_nodes [node_dim,node_num]
            % W_edges [edge_dim,edge_num]
            [W_nodes,W_edges] = get_graph_W (W(:,i_channel),config);
            
            phi_nodes_all{i_channel} = parts_features*W_nodes;
            phi_edges_all{i_channel} = parts_pair_features*W_edges;
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         channel_num = config.channel_num;
        %         save('debug_MRF_infer_traversal.mat','phi_nodes_all','phi_edges_all','channel_num');
        %         [scores,node_ass_idx] = MRF_infer_traversal_mex(phi_nodes_all,phi_edges_all,config.channel_num);
        
        
        % [scores,node_ass_idx] = MRF_infer_traversal_mex(phi_nodes_all,phi_edges_all,config.channel_num);
        for i_channel = 1:config.channel_num
            [max_score,node_ass_idx] = max(phi_nodes_all{i_channel},[],1);
            scores(i_channel) = sum(max_score);
        end
    otherwise
        fprintf('WRONG!!! the infer method not exist!\n');
        return;
end

[~,channel_max_score] = max(scores);

psi_max_score = zeros(1,config.graph.psi_dim*config.channel_num);
psi = [];
for i_node = 1:config.graph.node_num
    psi = [psi,parts_features(node_ass_idx(i_node),:)];
end
for p = 1:config.graph.node_num
    parts_id_p = node_ass_idx(p);
    for q = p+1:config.graph.node_num
        parts_id_q = node_ass_idx(q);
        
        min_id = min(parts_id_p,parts_id_q);
        max_id = max(parts_id_p,parts_id_q);
        
        psi = [psi, parts_pair_features(get_pair_ind(min_id,max_id,size(parts_features,1)),:)];
    end
end
psi_max_score( (channel_max_score-1)*config.graph.psi_dim+1:channel_max_score*config.graph.psi_dim )  = psi;

end


%%
function [W_nodes,W_edges] = get_graph_W (W,config)
W_nodes = reshape(W(1:config.graph.node_dim*config.graph.node_num),config.graph.node_dim,config.graph.node_num);

W_edges = reshape(W(1+config.graph.node_dim*config.graph.node_num:end),config.graph.edge_dim,config.graph.edge_num);

end


%%
function [kernel_constrait_D, u_D ,sum_mean_psi_pos_bags ]= compute_kernel_constrait_D(bag_datas_pos,config)

%--------------------------------------------------------
% compute the mean psi(X_i,P_i,z_i) of positive bags
%-------------------------------------------------------
pos_bags_num = size(bag_datas_pos,1);
feat_dim_each_channel = config.graph.psi_dim;
mean_psi_pos_bags = zeros(pos_bags_num,feat_dim_each_channel);

if config.using_parallel
    
    parfor i = 1:size(bag_datas_pos,1)
        fprintf('computing mean psi %d \n',i);
        mean_psi_pos_bags(i,:) = compute_bag_mean_psi(bag_datas_pos{i},config);
    end
    
else
    for i = 1:size(bag_datas_pos,1)
        fprintf('computing mean psi %d \n',i);
        mean_psi_pos_bags(i,:) = compute_bag_mean_psi(bag_datas_pos{i},config);
    end
end

sum_mean_psi_pos_bags = sum(mean_psi_pos_bags)/pos_bags_num;

%-----------------------------------
%compute constrait kernel
%-----------------------------------
u_D = zeros(feat_dim_each_channel*config.channel_num,config.channel_num^2);

for p = 1:config.channel_num
    for q = 1:config.channel_num
        
        u_D((p-1)*feat_dim_each_channel+1:(p)*feat_dim_each_channel ,config.channel_num*(p-1)+ q) = ...
            u_D((p-1)*feat_dim_each_channel+1:(p)*feat_dim_each_channel ,config.channel_num*(p-1)+ q) + sum_mean_psi_pos_bags';
        u_D((q-1)*feat_dim_each_channel+1:(q)*feat_dim_each_channel ,config.channel_num*(p-1)+ q) = ...
            u_D((q-1)*feat_dim_each_channel+1:(q)*feat_dim_each_channel ,config.channel_num*(p-1)+ q) - sum_mean_psi_pos_bags';
        
    end
end
u_D = u_D';
kernel_constrait_D = u_D*u_D';

end

%%

function mean_psi = compute_bag_mean_psi(bag_data,config)


mean_psi = zeros(1,config.graph.psi_dim);

parts_features = bag_data{1};
parts_pair_features = bag_data{2};

if (size(parts_features,2)+1) ~= config.graph.node_dim
    fprintf('WRONG!!! the dim of node wrong!\n');
    return;
end

if size(parts_pair_features,2) ~= config.graph.edge_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
    return;
end


% travel the hiddel space
parts_num = size(parts_features,1);

switch config.graph.node_num
    case 1
        for i = 1:parts_num
            mean_psi = mean_psi+parts_features;
        end
        mean_psi = mean_psi/parts_num;
    case 2
        
        for i = 1:parts_num^2
            
            [i_1,i_2] = ind2sub([parts_num,parts_num],i);
            mean_psi = mean_psi+[parts_features(i_1,:),1,parts_features(i_2,:),1,parts_pair_features(get_pair_ind(min(i_1,i_2),max(i_1,i_2),parts_num),:)];
        end
        mean_psi = mean_psi/(parts_num^2);
    case 3
        
        for i = 1:parts_num^3
            
            [i_1,i_2,i_3] = ind2sub([parts_num,parts_num,parts_num],i);
            mean_psi = mean_psi+[parts_features(i_1,:),parts_features(i_2,:),parts_features(i_3,:),...
                parts_pair_features(get_pair_ind(min(i_1,i_2),max(i_1,i_2),parts_num),:),...
                parts_pair_features(get_pair_ind(min(i_1,i_3),max(i_1,i_3),parts_num),:),...
                parts_pair_features(get_pair_ind(min(i_3,i_2),max(i_3,i_2),parts_num),:)];
        end
        mean_psi = mean_psi/(parts_num^3);
        
    otherwise
        fprintf('WRONG!!! do not support graph node_num > 3 !\n');
end


end

%%
function ind = get_pair_ind(i,j,n)
    ind = (2*n-i+2)*(i-1)/2 + j - i +1;
end







