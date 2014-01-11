%
% add response analyse
%

function return_state = m_responding_v2(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);
responding_config = global_config.responding;

% generate the splits and feature types according to global_config
splits = responding_config.splits;
feature_types = responding_config.feature_types;


global_config.learn_action_parts.struct_M5IL.graph.edge_num = global_config.learn_action_parts.struct_M5IL.graph.node_num*(global_config.learn_action_parts.struct_M5IL.graph.node_num-1)/2;
load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_pair.mat',1,1,1) );
load(load_name,'parts_pair_features');
global_config.learn_action_parts.struct_M5IL.graph.edge_dim = size(parts_pair_features,2);

load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',1,1,1,'shape') );
load(load_name,'parts_features');
global_config.learn_action_parts.struct_M5IL.graph.node_dim = size(parts_features,2)+1;

global_config.learn_action_parts.struct_M5IL.graph.psi_dim = global_config.learn_action_parts.struct_M5IL.graph.node_dim * global_config.learn_action_parts.struct_M5IL.graph.node_num + global_config.learn_action_parts.struct_M5IL.graph.edge_dim * global_config.learn_action_parts.struct_M5IL.graph.edge_num;

do_sets = global_config.coding.class_idx;

do_vids = cell(length(splits),class_num);
switch class(do_sets)
    case 'cell'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets{i}(1)} = [do_vids{cur_splits,do_sets{i}(1)};do_sets{i}(2)];
            end
            
        end
    case 'double'
        
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets(i)} = [train_set_idx{cur_splits}{do_sets(i)};test_set_idx{cur_splits}{do_sets(i)}];
            end
        end
        
    case 'char'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:class_num
                do_vids{cur_splits,i} = [train_set_idx{cur_splits}{i};test_set_idx{cur_splits}{i}];
            end
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% respoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    
    for i_feature = 1:length(feature_types)
        cur_feature = feature_types{i_feature};
        
        % loading the action part model
        W = {};
        for i_class = 1:class_num
            
            cur_class = i_class;
            if length(do_vids{cur_splits,cur_class}) == 0
                continue;
            end
            
            load_name = fullfile(global_config.learn_action_parts.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,cur_class,cur_feature));
            load( load_name, 'action_parts_model');
            
            W = [W,action_parts_model];
        end
        
%         if sum(sum(isnan(W)))
%            
%            fprintf('WRONG!!! NaN occur! splits:%d feature:%s\n ',cur_splits,cur_feature);
%            return_state = 0;
%            return;
%            
%         end
        
        for i_class = 1:class_num
            
            cur_class = i_class;
            if length(do_vids{cur_splits,cur_class}) == 0
                continue;
            end
            
            % all data
            fprintf('responding splits:%02d  c:%03d feature:%s\n',cur_splits,cur_class,cur_feature);
            
            cur_vid_idx = do_vids{cur_splits,cur_class};
            
                        
            if global_config.num_core > 1
                parfor i_vid = 1:length(cur_vid_idx)
                    do_responding(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),W,global_config.extract_part_features.path,responding_config.path,global_config.learn_action_parts.struct_M5IL);
                end
            else
                for i_vid = 1:length(cur_vid_idx)
                    do_responding(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),W,global_config.extract_part_features.path,responding_config.path,global_config.learn_action_parts.struct_M5IL);
                end
            end
            
            
            
        end  % iclass ends
    end  % ifeature_type
    
end % i_split

end

%%
function do_responding(cur_splits,cur_feature,cur_class,cur_vid,W,part_features_path,responding_path,config)

EPS = 1e-6;
% load instance feature data

load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
load(load_name,'parts_features');

parts_features = [parts_features,ones(size(parts_features,1) ,1)];

load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_pair.mat',cur_splits,cur_class,cur_vid) );
load(load_name,'parts_pair_features');

% responds_data = [parts_features,ones( size(parts_features,1) ,1)] * W;

channel_num = config.channel_num;

responds_data = zeros(length(W),channel_num);

for i = 1:length(W)
    
    cur_W = W{i};
    phi_nodes_all = {channel_num,1};
    phi_edges_all = {channel_num,1};
    for i_channel = 1:channel_num
        
        % W_nodes [node_dim,node_num]
        % W_edges [edge_dim,edge_num]
        [W_nodes,W_edges] = get_graph_W (cur_W(:,i_channel),config);
        
        phi_nodes_all{i_channel} = parts_features*W_nodes;
        phi_edges_all{i_channel} = parts_pair_features*W_edges;
        
    end
    
    responds_data(i,:) = infer_bag_score(phi_nodes_all,phi_edges_all,config);    
end



if sum(sum( isnan(responds_data) ))
    fprintf('WRONG!!! NaN occur in responsing! %s\n',load_name);
end

% reshape

responds_data = reshape(responds_data',1,[]);

save_name = fullfile(responding_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
save(save_name,'responds_data');


end

%%
function [W_nodes,W_edges] = get_graph_W (W,config)
W_nodes = reshape(W(1:config.graph.node_dim*config.graph.node_num),config.graph.node_dim,config.graph.node_num);

W_edges = reshape(W(1+config.graph.node_dim*config.graph.node_num:end),config.graph.edge_dim,config.graph.edge_num);

end

%%
% output: 
% scores [1,channel_num]
% psi_max_score, the related psi of the max score channel
function [scores,node_ass_idx] = infer_bag_score(phi_nodes_all,phi_edges_all,config)

switch config.method
    case 'TRW-S'
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [scores,node_ass_idx] = MRF_infer_TRW_S_mex(phi_nodes_all,phi_edges_all,config.channel_num);
        
    case 'traversal'
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         channel_num = config.channel_num;
%         save('debug_MRF_infer_traversal.mat','phi_nodes_all','phi_edges_all','channel_num');
        [scores,node_ass_idx] = MRF_infer_traversal_mex(phi_nodes_all,phi_edges_all,config.channel_num);
    case 'node-only'   
        
        scores = zeros(1,config.graph.node_num);
        node_ass_idx_all = zeros(config.channel_num,config.graph.node_num);
        for i_channel = 1:config.channel_num
            [max_val,node_ass_idx_all(i_channel,:)] = max( phi_nodes_all {i_channel} ,[],1);
            scores(i_channel) = sum(max_val);
        end
        [~,channel_max_score] = max(scores);
        node_ass_idx = node_ass_idx_all(channel_max_score,:);
        
    otherwise
        fprintf('WRONG!!! the infer method not exist!\n');
        return;
end



end
