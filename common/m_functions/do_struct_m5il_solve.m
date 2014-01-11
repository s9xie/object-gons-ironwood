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
function [model, info] = do_struct_m5il_solve(bag_paths_pos, bag_paths_neg, W_init, config)

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



%%%%%%%%%%%%%%%%%
% CCCP initial
%%%%%%%%%%%%%%%%

% init the kernel part of constrait D, need to travelling the hidden space of positive bags
[kernel_constrait_D,u_D] = compute_kernel_constrait_D(bag_paths_pos,config);

% init W_0 compute the initial energy, select the smallest initial one

energy_CCCP_rec = [];
energy_cutting_plane_all_rec = {};
energy_cutting_plane_end_rec = [];


if length(W_init) == 1
    % only one choise
    W = W_init{1};
    [energy,slack_neg,slack_pos,slack_div,balance_score,TPR,TNR,channel_distribute,subgradient_constrait_B,subgradient_constrait_C]  = compute_energy(bag_paths_pos,bag_paths_neg,W_init{1},config);
    
else
    energy = Inf;
    for i_init = 1:length(W_init)
        [energy_i,slack_neg_i,slack_pos_i,slack_div_i,balance_score_i,TPR_i,TNR_i,channel_distribute_i,subgradient_constrait_B_i,subgradient_constrait_C_i] = compute_energy(bag_paths_pos,bag_paths_neg,W_init{i_init},config);
        if energy_i < energy
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
    
end
energy_CCCP_rec = [energy_CCCP_rec;energy];

fprintf('init...\n CCCP:  slack_neg = %.2f, slack_pos = %.2f, slack_div = %.2f, balance_score = %.2f, TPR = %.2f, TNR = %.2f, channel_distribute = ', ...
    slack_neg, slack_pos,    slack_div,    balance_score, TPR, TNR); disp(channel_distribute);

%%%%%%%%%%%%%%%%%
% CCCP main loop
%%%%%%%%%%%%%%%%%
iter_CCCP = 0;
while true
    iter_CCCP = iter_CCCP +1;
    fprintf('CCCP iter:%d: computing subgradient...\n',iter_CCCP);
    W0 = W;
    
    %--------------------
    % cutting plane
    %--------------------
    [W, energy_cutting_plane_dual, energy_cutting_plane_primal, slack_neg,slack_pos,slack_div] = ...
        solve_cuttingplane( W, subgradient_constrait_B,subgradient_constrait_C, kernel_constrait_D, u_D, bag_paths_pos, bag_paths_neg, config );
    
    energy_cutting_plane_all_rec = [ energy_cutting_plane_all_rec; energy_cutting_plane_primal ];
    energy_cutting_plane_end_rec = [energy_cutting_plane_end_rec; energy_cutting_plane_primal(end)];
    fprintf(' energy cutting plane= %.2f, energy CCCP = %.2f:\n', energy_cutting_plane_primal(end), energy_CCCP_rec(end) );
    fprintf(' slack_neg = %.2f, slack_pos = %.2f, slack_div = %.2f;\n', slack_neg,slack_pos,slack_div );
    
    
    %---------------------------------------------------
    % computing subgradient for constrait B and C
    %---------------------------------------------------
    [energy_CCCP,slack_neg,slack_pos,slack_div,balance_score,TPR,TNR,channel_distribute,subgradient_constrait_B,subgradient_constrait_C] = compute_energy(bag_paths_pos,bag_paths_neg,W,config);
    energy_CCCP_rec = [energy_CCCP_rec; energy_CCCP];
    
    fprintf(' CCCP:  slack_neg = %.2f, slack_pos = %.2f, slack_div = %.2f, balance_score = %.2f, TPR = %.2f, TNR = %.2f, channel_distribute = ', ...
        slack_neg, slack_pos,    slack_div,    balance_score, TPR, TNR); disp(channel_distribute);
    
    count_iter_cccp = count_iter_cccp + 1;
    
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
        if sqrt( sum((W(:)-W0(:)).^2) ) <= config.stop.minDetaW
            fprintf(' achieving minimum deta of L2-norm distance of model parameters;\n');
            break; % flag_convg = true;
        end
    end
    
    if any( strcmp( 'numIterCCCP', config.stop.criterion ) )
        if count_iter_cccp >= config.stop.maxNumIterCCCP
            fprintf(' achieving maximum number of CCCP iterations;\n');
            break; % flag_convg = true;
        end
    end
    
end % main loop of CCCP

model = W;
info.energy = energy_CCCP;

end % function end: do_struct_m5il_solve

%%
function [W, energy_cutting_plane_dual, energy_cutting_plane_primal, slack_neg,slack_pos,slack_div] = solve_cuttingplane( W, subgradient_constrait_B,subgradient_constrait_C, kerMat_ws, uMat_ws, bag_paths_pos,bag_paths_neg, config )

DEBUG = false;

bag_pos_num = length(bag_paths_pos);
bag_pos_neg = length(bag_paths_neg);

C = [config.C_neg;config.C_pos,config.C_div];
zeta = config.zeta;
epsilon = config.eps;
siz_incremt_ws = config.siz_incremt_ws;

slack = zeros(3,1);

energy_cutting_plane_dual = [];
energy_cutting_plane_primal = [];
                   
W_vec = W(:)';
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
            value_vlt_new = find_most_vlt_const_neg( wVec, bag_paths_neg , config );
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
    [lamVec, loss_dual_iter] = solve_qp_dual_v1( kerMat_ws(1:count_const_ws,1:count_const_ws), bVec_ws, indices_const_ws, C, Params.solveQP );
    energy_cutting_plane_dual = [energy_cutting_plane_dual; loss_dual_iter];
    
    % update <W> and slack variables;
    W_vec = uMat_ws*lamVec;
    W = reshape(W_vec,[dim_feat num_clust]);
    W_vec = W_vec';
    for j = 1: length(slack)
        if ~isempty(indices_const_ws{j})
            slack(j) = max( max( bVec_ws(indices_const_ws{j})' - W_vec*uMat_ws(:,indices_const_ws{j}) ), 0 );
        end
    end    
    % compute primal loss;
    energy_primal_iter = (W_vec*W_vec')/2;
    for j = 1: length(slack)
        energy_primal_iter = energy_primal_iter + C(j)*slack(j);
    end
    energy_cutting_plane_primal = [energy_cutting_plane_primal; energy_primal_iter];
    if abs(energy_primal_iter-loss_dual_iter)>epsilon
        if DEBUG
            fprintf('Warning: the difference loss (%f) of primal and dual is larger than epsilon (%f);\n', ...
                     abs(energy_primal_iter-loss_dual_iter), epsilon);
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
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_neg( wVec, bag_paths_neg , config );
                
            case 2
                % (2) for constraint B:                
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_pos( W_vec, subgradient_constrait_B, bag_pos_num );
                
            case 3
                % (3) for constraint C:
                [value_vlt_new, uVec_new, b_new] = find_most_vlt_const_pos( W_vec, subgradient_constrait_C, bag_pos_num );
                
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
            energy_cutting_plane_dual = [energy_cutting_plane_dual; loss_dual_iter];
            
            % update <W> and slack variables;
            W_vec = uMat_ws*lamVec;
            W = reshape(W_vec,[dim_feat num_clust]);
            W_vec = W_vec';
            for j = 1: length(slack)
                if ~isempty(indices_const_ws{j})
                    slack(j) = max( max( bVec_ws(indices_const_ws{j})' - W_vec*uMat_ws(:,indices_const_ws{j}) ), 0 );
                end
            end
            
            % compute primal loss;
            energy_primal_iter = (W_vec*W_vec')/2;
            for j = 1: length(slack)
                energy_primal_iter = energy_primal_iter + C(j)*slack(j);
            end
            energy_cutting_plane_primal = [energy_cutting_plane_primal; energy_primal_iter];
            if abs(energy_primal_iter-loss_dual_iter)>epsilon
                if DEBUG
                    fprintf('Warning: the difference loss (%f) of primal and dual is larger than epsilon (%f);\n', ...
                     abs(energy_primal_iter-loss_dual_iter), epsilon);
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

end


%% find the most violated constraint for (B) or (C);
function [value_vlt, uVec, b] = find_most_vlt_const_pos( wVec, uMat, N )
indVec_const_violated = ( uMat*wVec < 1 );
uVec = sum( uMat(:,indVec_const_violated), 2 )/N;
b = sum(indVec_const_violated)/N; 
value_vlt = b - wVec*uVec;
end % end of function;

%% find the most violated constraint for (A);
function [value_vlt, uVec, b] = find_most_vlt_const_neg(wVec, bag_paths_neg , config )

bag_num_neg = length(bag_paths_neg);

scores_bag_neg = zeros(bag_num_neg,config.channel_num);
psi_bag_neg = cell(bag_num_neg,1);
graph_struct_neg = cell(bag_num_neg,1);
if config.using_parallel
    
    parfor i = 1:bag_num_neg
        [scores_bag_neg(i,:), psi_bag_neg{i}, graph_struct_neg{i}] = infer_max_psi(bag_paths_neg{i},W,config);
    end
else
    for i = 1:bag_num_neg
        [scores_bag_neg(i,:), psi_bag_neg{i}, graph_struct_neg{i}] = infer_max_psi(bag_paths_neg{i},W,config);
    end
    
end

scoreVec_max_bag_neg = max(scores_bag_neg,[],2);

indVec_bag_violated = ( scoreVec_max_bag_neg > -1 );

uVec = -sum(cell2mat(psi_bag_neg),1)/num_bag_neg;

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

%%  compute the energy, given data and model
function [energy,slack_neg,slack_pos,slack_div,balance_score,TPR,TNR,channel_distribute,subgradient_constrait_B,subgradient_constrait_C] = compute_energy(bag_paths_pos,bag_paths_neg,W,config)

energy = sum(sum(W.*W));

% check constrait D
pos_bags_num = length(bag_paths_pos);
feat_dim_each_channel = config.graph.psi_dim;
mean_psi_pos_bags = zeros(pos_bags_num,feat_dim_each_channel);

if config.using_parallel
    
    parfor i = 1:length(bag_idx)
        mean_psi_pos_bags(i,:) = compute_bag_mean_psi(bag_paths_pos{i},config);
    end
    
else
    for i = 1:length(bag_idx)
        mean_psi_pos_bags(i,:) = compute_bag_mean_psi(bag_paths_pos{i},config);
    end
end

sum_mean_psi_pos_bags = sum(mean_psi_pos_bags)/pos_bags_num;
channel_scores = sum_mean_psi_pos_bags*W;

balance_score = 0;
for i_channel = 1:config.channel_num
    for q = i_channel+1:config.channel_num
        
        balance_score = balance_score + abs(channel_scores(i_channel) - channel_scores(q));
        if abs(channel_scores(i_channel) - channel_scores(q)) > config.zeta
            fprintf('ERROR!!! the channel balance constraint not satisfied!\n');
        end
        
    end
end

%--------- constrait A -------------
bag_num_neg = length(bag_paths_neg);

scores_bag_neg = zeros(bag_num_neg,config.channel_num);
psi_bag_neg = cell(bag_num_neg,1);
graph_struct_neg = cell(bag_num_neg,1);
if config.using_parallel
    
    parfor i = 1:bag_num_neg
        [scores_bag_neg(i,:), psi_bag_neg{i}, graph_struct_neg{i}] = infer_max_psi(bag_paths_neg{i},W,config);
    end
else
    for i = 1:bag_num_neg
        [scores_bag_neg(i,:), psi_bag_neg{i}, graph_struct_neg{i}] = infer_max_psi(bag_paths_neg{i},W,config);
    end
    
end


slack_constrait_A = max(scores_bag_neg,[],2);
slack_constrait_A = max( (1+slack_constrait_A), 0 );

slack_neg = sum(slack_constrait_A)/ bag_num_neg;
energy = energy + config.C_neg * slack_neg;

%--------- constrait B & C --------------
bag_num_pos = length(bag_paths_pos);

scores_bag_pos = zeros(bag_num_pos,config.channel_num);
psi_bag_pos = cell(bag_num_pos,1);
graph_struct_pos = cell(bag_num_pos,1);

diversities = zeros(bag_num_pos,1);
psi_bag_div= cell(bag_num_pos,1);
graph_struct_div = cell(bag_num_pos,1);

if config.using_parallel
    
    parfor i = 1:bag_num_pos
        [scores_bag_pos(i,:), psi_bag_pos{i}, graph_struct_pos{i},diversities(i),psi_bag_div{i},graph_struct_div{i}] = infer_max_psi_div(bag_paths_pos{i},W,config);
    end
else
    for i = 1:bag_num_pos
        [scores_bag_pos(i,:), psi_bag_pos{i}, graph_struct_pos{i},diversities(i),psi_bag_div{i},graph_struct_div{i}] = infer_max_psi_div(bag_paths_pos{i},W,config);
    end
    
end


subgradient_constrait_B = cell2mat(psi_bag_pos);
subgradient_constrait_C = cell2mat(psi_bag_div);


slack_constrait_B = max(scores_bag_pos,[],2);
slack_constrait_B = max( (1-slack_constrait_B), 0 );
slack_pos = sum(slack_constrait_B)/ bag_num_pos;
energy = energy + config.C_pos *slack_pos;

slack_constrait_C = max( (1-diversities) , 0);
slack_div = sum(slack_constrait_C)/ bag_num_pos;
energy = energy + config.C_div *slack_div;

% the max score channels of positive bags
TPR = sum(max(scores_bag_pos,[],2) > 0) / bag_num_pos;
TNR = sum(max(scores_bag_neg,[],2) < 0) / bag_num_neg;

channel_distribute = zeros(config.channel_num,1);

[scores_bag_pos_max, channels_bags_pos] = max( scores_bag_neg, [], 2 );
channels_bags_pos = channels_bags_pos(scores_bag_pos_max>0);

for i_channel = 1: config.channel_num
    channel_distribute(i_channel) = sum(channels_bags_pos==i_channel);
end % p: index of cluster;
channel_distribute = channel_distribute / sum(channels_bags_pos);

end

%%
% for positive bags, there are two constraint, using travelsal to get the
% max score and also diversity
function [scores, psi_max_score, graph_struct_max_score,diversities, psi_max_div, graph_struct_max_div] = infer_max_psi_div(bag_path,W,config)

scores = zeros(1,config.channel_num);
psi_max_score = zeros(1,config.psi_dim*config.channel_num);
graph_struct_max_score = zeros(1,config.node_num);

diversities = 0;
psi_max_div = zeros(1,config.psi_dim*config.channel_num);
graph_struct_max_div = zeros(1,config.node_num);

% load bag data
if ~exist(bag_path,'file')
    fprintf('WRONG!!! bag data not exist! %s\n',bag_path);
    return;
end
load(bag_path,'parts_features','parts_pair_features');

if size(parts_features,1) ~= config.graph.node_dim
    fprintf('WRONG!!! the dim of node wrong!\n');
end


if size(parts_pair_features,1) ~= config.graph.edge_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
end

% put the parts_pair_features from cell to mat
parts_num = size(parts_features,1);
parts_pair_features_mat = zeros(parts_num*(parts_num+1)/2,config.graph.edge_dim);
edge_nodes = zeros(parts_num*(parts_num+1)/2,2);
row_ind = 0;
for i = 1:parts_num
    for j = i:parts_num
        row_ind = row_ind +1;
        edge_nodes(row_ind,:) = [i,j];
        parts_pair_features_mat(row_ind,:) = parts_pair_features{i,j};
    end
end


phi_nodes_all = cell(config.channel_num,1);
phi_edges_all = cell(config.channel_num,1);
for i_channel = 1:config.channel_num
    
    % W_nodes [node_dim,node_num]
    % W_edges [edge_dim,edge_num]
    [W_nodes,W_edges] = get_graph_W (W(:,i_channel),config);
    
    phi_nodes_all{i_channel} = parts_features*W_nodes;
    phi_edges_all{i_channel} = parts_pair_features_mat*W_edges;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[scores, graph_struct_max_score,diversities , channel_max_div, graph_struct_max_div] =...
    MRF_infer_div_trasveral_mex(phi_nodes_all,phi_edges_all,config.channel_num);

% generate the psi

[~,channel_max_score] = max(scores);
psi = [];
for i_node = 1:config.node_num
    psi = [psi,parts_features(graph_struct_max_score(i_node),:)];  
end
for p = 1:config.node_num
    parts_id_p = graph_struct_max_score(p);
    for q = p+1:config.node_num
        parts_id_q = graph_struct_max_score(q);
        
        min_id = min(parts_id_p,parts_id_q);
        max_id = max(parts_id_p,parts_id_q);
        
        psi = [psi, parts_pair_features{min_id,max_id}];    
    end
end

psi_max_score( (channel_max_score-1)*config.psi_dim+1:channel_max_score*config.psi_dim )  = psi;




psi = [];
max_channel_graph_struct = graph_struct_max_div(:,channel_max_div);
for i_node = 1:config.node_num
    psi = [psi,parts_features(max_channel_graph_struct(i_node),:)];
end

for p = 1:config.node_num
    parts_id_p = max_channel_graph_struct(p);
    for q = p+1:config.node_num
        parts_id_q = max_channel_graph_struct(q);
        
        min_id = min(parts_id_p,parts_id_q);
        max_id = max(parts_id_p,parts_id_q);
        
        psi = [psi, parts_pair_features{min_id,max_id}];
    end
end
psi_max_div( (channel_max_div-1)*config.psi_dim+1:channel_max_div*config.psi_dim )  = psi;

for i_channel = 1:config.channel_num
   
    psi = [];
    for i_node = 1:config.node_num
        psi = [psi,parts_features(graph_struct_max_div(i_node,i_channel),:)];
    end

    for p = 1:config.node_num
        parts_id_p = graph_struct_max_div(p,i_channel);
        for q = p+1:config.node_num
            parts_id_q = graph_struct_max_div(q,i_channel);
            
            min_id = min(parts_id_p,parts_id_q);
            max_id = max(parts_id_p,parts_id_q);
            
            psi = [psi, parts_pair_features{min_id,max_id}];
        end
    end
    psi_max_div( (channel_max_div-1)*config.psi_dim+1:channel_max_div*config.psi_dim )  = ...
        psi_max_div( (channel_max_div-1)*config.psi_dim+1:channel_max_div*config.psi_dim ) - psi/config.channel_num;
    
end

psi_max_div = psi_max_div*config.channel_num/(config.channel_num-1);

end


%% the function to infer the max score of bag
% input: bag_path, W [psi_dim,config.channel_num], config
function [scores, psi_max_score, graph_struct_max_score] = infer_max_psi(bag_path,W,config)

scores = zeros(1,config.channel_num);
graph_struct = cell(config.channel,1);

% load bag data
if ~exist(bag_path,'file')
    fprintf('WRONG!!! bag data not exist! %s\n',bag_path);
    return;
end
load(bag_path,'parts_features','parts_pair_features');

if size(parts_features,1) ~= config.graph.node_dim
    fprintf('WRONG!!! the dim of node wrong!\n');
end


if size(parts_pair_features,1) ~= config.graph.edge_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
end

% put the parts_pair_features from cell to mat
parts_num = size(parts_features,1);
parts_pair_features_mat = zeros(parts_num*(parts_num+1)/2,config.graph.edge_dim);
edge_nodes = zeros(parts_num*(parts_num+1)/2,2);
row_ind = 0;
for i = 1:parts_num
    for j = i:parts_num
        row_ind = row_ind +1;
        edge_nodes(row_ind,:) = [i,j];
        parts_pair_features_mat(row_ind,:) = parts_pair_features{i,j};
    end
end



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
            phi_edges_all{i_channel} = -parts_pair_features_mat*W_edges;
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [scores,graph_struct] = MRF_infer_TRW_S_mex(phi_nodes_all,phi_edges_all,config.channel_num);
      
    case 'traversal'
        
        phi_nodes_all = {config.channel_num,1};
        phi_edges_all = {config.channel_num,1};
        for i_channel = 1:config.channel_num
            
            % W_nodes [node_dim,node_num]
            % W_edges [edge_dim,edge_num]
            [W_nodes,W_edges] = get_graph_W (W(:,i_channel),config);
            
            phi_nodes_all{i_channel} = parts_features*W_nodes;
            phi_edges_all{i_channel} = parts_pair_features_mat*W_edges;
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [scores,graph_struct] = MRF_infer_trasveral_mex(phi_nodes_all,phi_edges_all,config.channel_num);
        
    otherwise
        fprintf('WRONG!!! the infer method not exist!\n');
        return;
end

[~,channel_max_score] = max(scores);
graph_struct_max_score = graph_struct{channel_max_score};

psi = [];
for i_node = 1:config.graph.node_num
    psi = [psi,parts_features(graph_struct_max_score(i_node),:)];  
end
for p = 1:config.graph.node_num
    parts_id_p = graph_struct_max_score(p);
    for q = p+1:config.graph.node_num
        parts_id_q = graph_struct_max_score(q);
        
        min_id = min(parts_id_p,parts_id_q);
        max_id = max(parts_id_p,parts_id_q);
        
        psi = [psi, parts_pair_features{min_id,max_id}];    
    end
end
psi_max_score( (channel_max_score-1)*config.psi_dim+1:channel_max_score*config.psi_dim )  = psi;

end


%%
function [W_nodes,W_edges] = get_graph_W (W,config)
W_nodes = reshape(W(1:config.node_dim*config.graph.node_num),config.node_dim,config.graph.node_num);

W_edges = reshape(W(1+config.node_dim*config.graph.node_num:end),config.edge_dim,config.edge_num);

end


%%
function [kernel_constrait_D, u_D]= compute_kernel_constrait_D(bag_paths_pos,config)

%--------------------------------------------------------
% compute the mean psi(X_i,P_i,z_i) of positive bags
%-------------------------------------------------------
pos_bags_num = length(bag_paths_pos);
feat_dim_each_channel = config.graph.psi_dim;
mean_psi_pos_bags = zeros(pos_bags_num,feat_dim_each_channel);

if config.using_parallel
    
    parfor i = 1:length(bag_idx)
        mean_psi_pos_bags(i,:) = compute_bag_mean_psi(bag_paths_pos{i},config);
    end
    
else
    for i = 1:length(bag_idx)
        mean_psi_pos_bags(i,:) = compute_bag_mean_psi(bag_paths_pos{i},config);
    end
end

sum_mean_psi_pos_bags = sum(mean_psi_pos_bags)/pos_bags_num;

%-----------------------------------
%compute constrait kernel
%-----------------------------------
u_D = zeros(feat_dim_each_channel*config.channel_num,config.channel_num^2);

for p = 1:config.channel_num
    for q = 1:config.channel_num
        
        u_D((p-1)*feat_dim_each_channel:(p)*feat_dim_each_channel ,config.channel_num*(p-1)+ q) = sum_mean_psi_pos_bags;
        u_D((q-1)*feat_dim_each_channel:(q)*feat_dim_each_channel ,config.channel_num*(p-1)+ q) = -sum_mean_psi_pos_bags;
        
    end
end

kernel_constrait_D = u_D'*u_D;

end

%%

function mean_psi = compute_bag_mean_psi(bag_path,config)


mean_psi = zeros(1,config.graph.psi_dim);
% load bag_data
if ~exist(bag_path,'file')
    fprintf('WRONG!!! bag data not exist! %s\n',bag_path);
    return;
end
load(bag_path,'parts_features','parts_pair_features');

if size(parts_features,1) ~= config.graph.node_dim
    fprintf('WRONG!!! the dim of node wrong!\n');
end


if size(parts_pair_features,1) ~= config.graph.edge_dim
    fprintf('WRONG!!! the dim of edge wrong!\n');
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
            mean_psi = mean_psi+[parts_features(i_1,:),parts_features(i_2,:),parts_pair_features{min(i_1,i_2),max(i_1,i_2)}];
        end
        mean_psi = mean_psi/(parts_num^2);
    case 3
        
        for i = 1:parts_num^3
            
            [i_1,i_2,i_3] = ind2sub([parts_num,parts_num,parts_num],i);
            mean_psi = mean_psi+[parts_features(i_1,:),parts_features(i_2,:),parts_features(i_3,:),parts_pair_features{min(i_1,i_2),max(i_1,i_2)},parts_pair_features{min(i_1,i_3),max(i_1,i_3)},parts_pair_features{min(i_2,i_3),max(i_2,i_3)}];
        end
        mean_psi = mean_psi/(parts_num^3);
        
    otherwise
        fprintf('WRONG!!! do not support graph node_num > 3 !\n');
end


end






