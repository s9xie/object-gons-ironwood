
%v1 support hog-1,hog-2
%v2 add x-2 SPM this version
%v3 merge the responding and extract_mid_level part

function return_state = m_extract_mid_level_descriptors_v3_vc(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

emld_config = global_config.extract_mid_level_descriptors;
splits = emld_config.splits;
feature_types = emld_config.feature_types;

%get the info of graph
global_config.learn_action_parts.struct_M5IL.graph.edge_num = global_config.learn_action_parts.struct_M5IL.graph.node_num*(global_config.learn_action_parts.struct_M5IL.graph.node_num-1)/2;
load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_pair.mat',1,1,1) );
load(load_name,'parts_pair_features');
global_config.learn_action_parts.struct_M5IL.graph.edge_dim = size(parts_pair_features,2);

load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',1,1,1,'vc') );
load(load_name,'parts_features');
global_config.learn_action_parts.struct_M5IL.graph.node_dim = size(parts_features,2)+1;

global_config.learn_action_parts.struct_M5IL.graph.psi_dim = global_config.learn_action_parts.struct_M5IL.graph.node_dim * global_config.learn_action_parts.struct_M5IL.graph.node_num + global_config.learn_action_parts.struct_M5IL.graph.edge_dim * global_config.learn_action_parts.struct_M5IL.graph.edge_num;


% specify the videos to be processed
do_sets = emld_config.class_idx;

do_imgs = cell(length(splits),class_num);
switch class(do_sets)
    case 'cell'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_imgs{cur_splits,do_sets{i}(1)} = [do_imgs{cur_splits,do_sets{i}(1)};do_sets{i}(2)];
            end
            
        end
    case 'double'
        
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_imgs{cur_splits,do_sets(i)} = [train_set_idx{cur_splits}{do_sets(i)} test_set_idx{cur_splits}{do_sets(i)}]';
            end
        end
        
    case 'char'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:class_num
                do_imgs{cur_splits,i} = [train_set_idx{cur_splits}{i};test_set_idx{cur_splits}{i}];
            end
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end


for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    for i_feature = 1:length(feature_types)
        cur_feature = feature_types{i_feature};
        
        % load the W of models. graph with one node and more than one nodes
        % are different
        switch cur_feature(end)
            
            case '1'
                W = [];
                for i_class = 1:class_num
                    
                    cur_class = i_class;
                    if length(do_imgs{cur_splits,cur_class}) == 0
                        continue;
                    end
                    
                    load_name = fullfile(global_config.learn_action_parts.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,cur_class,cur_feature));
                    load( load_name, 'action_parts_model');
                    
                    W = [W,action_parts_model];
                end
            otherwise
                W = {};
                for i_class = 1:class_num
                    
                    cur_class = i_class;
                    if length(do_imgs{cur_splits,cur_class}) == 0
                        continue;
                    end
                    
                    load_name = fullfile(global_config.learn_action_parts.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,cur_class,cur_feature));
                    load( load_name, 'action_parts_model');
                    
                    W = [W,action_parts_model];
                end
%             otherwise
%                 fprintf('WRONG!!! wrong feature type %s\n',cur_feature);
%                 return;
        end
        
        
        for i_class = 1:class_num
            
            cur_class = i_class;
            if isempty(do_imgs{cur_splits,cur_class})
                continue;
            end
            
            fprintf('extract mid level split:%d class:%d feature:%s\n',cur_splits,cur_class,cur_feature);
            % all data
            cur_vid_idx = do_imgs{cur_splits,i_class};
            
            switch emld_config.params.method
                case 'max'
                    % do struct pooling
                    if global_config.num_core > 1
                        parfor i_vid = 1:length(cur_vid_idx)
                            do_struct_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),W,global_config);
                        end
                    else
                        for i_vid = 1:length(cur_vid_idx)
                            do_struct_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),W,global_config);
                        end
                    end
                    
                otherwise
                    fprintf('wrong type of elld_config.params.method %s!\n',emld_config.params.method);
                    return_state = 0;
                    return;
            end
            
        end  % ifeature_type
    end % iclass ends
    
end % i_split



end

%%
function do_struct_pooling_max(cur_splits,cur_feature,cur_class,cur_vid,W,global_config)
% for every cell, get the response of all the action gons

emld_config = global_config.extract_mid_level_descriptors;
feature_path = global_config.extract_features.path;
part_features_path = global_config.extract_part_features.path;
struct_M5IL_config = global_config.learn_action_parts.struct_M5IL;

% fprintf('coding split:%d class:%d vid:%d feature:%s\n',cur_splits,cur_class,cur_vid,cur_feature);

%load_name = fullfile(feature_path,sprintf('c%03d_v%03d_info.mat',cur_class,cur_vid));
%load(load_name,'vid_info');

cur_part_feature = cur_feature(1:end-2);

load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_part_feature) );
load(load_name,'parts_features');

parts_features = [parts_features,ones(size(parts_features,1) ,1)];


% load instance feature data
load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_location.mat',cur_splits,cur_class,cur_vid) );
load(load_name,'parts_location');

responds_location  = parts_location;
% switch emld_config.params.pooling_range_type
%     
%     case 'video'
%         pooling_range = [vid_info(1),vid_info(1),vid_info(2),vid_info(2),vid_info(3),vid_info(3)];
%         pooling_cells = emld_config.params.pooling_cells.*...
%             (repmat(pooling_range,size(emld_config.params.pooling_cells,1),1));
%         
%    case 'bbox'
% min_loc = min(responds_location(:,1:3),[],1);
% delta_loc = max(responds_location(:,1:3),[],1) - min_loc;
% 
% low_range = [min_loc(2),min_loc(2),min_loc(3),min_loc(3),min_loc(1),min_loc(1)];
% delta_range = [delta_loc(2),delta_loc(2),delta_loc(3),delta_loc(3),delta_loc(1),delta_loc(1)];
% 
% pooling_cells = repmat(low_range,size(emld_config.params.pooling_cells,1),1) + ...
%     emld_config.params.pooling_cells.* repmat(delta_range,size(emld_config.params.pooling_cells,1),1);
%         
%     otherwise
%         fprintf('WRONG!!! the type of pooling range %s not exist!\n ',emld_config.params.pooling_range);
%         
% end

min_loc = min(responds_location(:,1:2),[],1);
delta_loc = max(responds_location(:,1:2),[],1) - min_loc;

low_range = [min_loc(1),min_loc(1),min_loc(2),min_loc(2)];
delta_range = [delta_loc(1),delta_loc(1),delta_loc(2),delta_loc(2)];
%%DEBUG!!!CHECK SIZE
pooling_cells = repmat(low_range,size(emld_config.params.pooling_cells,1),1) + ...
    emld_config.params.pooling_cells.* repmat(delta_range,size(emld_config.params.pooling_cells,1),1);

pooling_cells_num = size(pooling_cells,1);

switch cur_feature(end)
    
    case '1' %for graph with one node
        
        responds_data = parts_features * W;
        if sum(sum( isnan(responds_data) ))
            fprintf('WRONG!!! NaN occur in responsing! %s\n',load_name);
        end
        
        
        % pooling --------------------
        ml_descriptor = [];
        for i_cell = 1:pooling_cells_num
            
            select_id = (responds_location(:,2) >= pooling_cells(i_cell,1))&(responds_location(:,2) <= pooling_cells(i_cell,2))&...
                (responds_location(:,3) >= pooling_cells(i_cell,3))&(responds_location(:,3) <= pooling_cells(i_cell,4))&...
                (responds_location(:,1) >= pooling_cells(i_cell,5))&(responds_location(:,1) <= pooling_cells(i_cell,6));
            
            responds_in_cell = responds_data(find(select_id == 1),:);
            if(size(responds_in_cell,1) > 0 )
                ml_descriptor = [ml_descriptor,( max(responds_in_cell,[],1))];
            else
                ml_descriptor = [ml_descriptor,zeros(1,size(responds_in_cell,2))];
            end
        end
        
        if sum(sum(isnan(ml_descriptor)))
            printf('WRONG!! NaN occur!\n');
        end
        
        
        
        save_name = fullfile(emld_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
        save(save_name,'ml_descriptor');
        
        
    otherwise %for graph with more than one node
        
       
        load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_pair.mat',cur_splits,cur_class,cur_vid) );
        load(load_name,'parts_pair_features');
        
        channel_num = struct_M5IL_config.channel_num;
        
        % compute all the scores of nodes and edges 
        phi_nodes_all = cell(channel_num,length(W));
        phi_edges_all = cell(channel_num,length(W));
        
        for i = 1:length(W)
            
            cur_W = W{i};
            
            for i_channel = 1:channel_num
                
                % W_nodes [node_dim,node_num]
                % W_edges [edge_dim,edge_num]
                [W_nodes,W_edges] = get_graph_W (cur_W(:,i_channel),struct_M5IL_config);
                
                phi_nodes_all{i_channel,i} = parts_features*W_nodes;
                phi_edges_all{i_channel,i} = parts_pair_features*W_edges;
                
            end
            
            %             responds_data(i,:) = infer_bag_score(phi_nodes_all,phi_edges_all,config);
        end
        
        channel_num = size(phi_nodes_all,1);
        W_num = size(phi_nodes_all,2);
        parts_num = size(phi_nodes_all{1,1},1);
        
        
        % pooling --------------------
        ml_descriptor = [];
        for i_cell = 1:pooling_cells_num
            
            % get the parts in current cell
            select_id = (responds_location(:,1) >= pooling_cells(i_cell,1))&(responds_location(:,1) <= pooling_cells(i_cell,2))&...
                (responds_location(:,2) >= pooling_cells(i_cell,3))&(responds_location(:,2) <= pooling_cells(i_cell,4));
            
            select_nodes_id = find(select_id == 1);
            select_edges_id = [];
            for i = 1:length(select_nodes_id)
                
                cur_i = select_nodes_id(i);
                for j = i:length(select_nodes_id)
                    cur_j = select_nodes_id(j);
                    cur_id =  (2*parts_num-cur_i+2)*(cur_i-1)/2 + cur_j - cur_i +1;
                    select_edges_id = [select_edges_id;cur_id];
                    
                end
                
            end
            
            % compute the reponse of cur cell
            responds_data = zeros(W_num,channel_num);
            
            if(~isempty(select_nodes_id))
                
                for i_W = 1:size(phi_nodes_all,2)
                    
                    phi_nodes_in_cell = cell(channel_num,1);
                    phi_edges_in_cell = cell(channel_num,1);
                    for i_channel = 1:channel_num
                        
                        phi_nodes_in_cell{i_channel} = phi_nodes_all{i_channel,i_W}(select_nodes_id,:);
                        phi_edges_in_cell{i_channel} = phi_edges_all{i_channel,i_W}(select_edges_id,:);
                        
                    end
                     % infer in the cell
                    responds_data(i_W,:) = infer_bag_score(phi_nodes_in_cell,phi_edges_in_cell,struct_M5IL_config);
                end
                
            end
            
            responds_data = reshape(responds_data',1,[]);
            ml_descriptor = [ml_descriptor,responds_data];
            
        end
        
        if sum(sum(isnan(ml_descriptor)))
            fprintf('WRONG!! NaN occur!\n');
        end
        
        
        save_name = fullfile(emld_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
        save(save_name,'ml_descriptor');
        
        
end


end


%%
function [W_nodes,W_edges] = get_graph_W (W,config)
W_nodes = reshape(W(1:config.graph.node_dim*config.graph.node_num),config.graph.node_dim,config.graph.node_num);

W_edges = reshape(W(1+config.graph.node_dim*config.graph.node_num:end),config.graph.edge_dim,config.graph.edge_num);

end

%%
% infer the scores in each cell
function [scores,node_ass_idx] = infer_bag_score(phi_nodes_all,phi_edges_all,config)

switch config.method
    case 'TRW-S'
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         for i_channel = 1:config.channel_num
            phi_nodes_all {i_channel} = -phi_nodes_all {i_channel};
            phi_edges_all {i_channel} = -phi_edges_all {i_channel};
            
        end
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