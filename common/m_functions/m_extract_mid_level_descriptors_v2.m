
%v1 support hog-1,hog-2
%v2 add x-2 SPM this version
function return_state = m_extract_mid_level_descriptors_v2(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

emld_config = global_config.extract_mid_level_descriptors;
splits = emld_config.splits;
feature_types = emld_config.feature_types;


do_sets = emld_config.class_idx;

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


for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    for i_class = 1:class_num
        
        cur_class = i_class;
        if length(do_vids{cur_splits,cur_class}) == 0
            continue;
        end
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};
            fprintf('extract mid level split:%d class:%d feature:%s\n',cur_splits,cur_class,cur_feature);
            % all data
            cur_vid_idx = do_vids{cur_splits,i_class};
            
            switch emld_config.params.method
                case 'max'
                    %         do_pooling_max
                    if global_config.num_core > 1
                        parfor i_vid = 1:length(cur_vid_idx)
                            do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),emld_config,global_config.responding.path,global_config.extract_features.path,global_config.learn_action_parts.struct_M5IL);
                        end
                    else
                        for i_vid = 1:length(cur_vid_idx)
                            do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),emld_config,global_config.responding.path,global_config.extract_features.path,global_config.learn_action_parts.struct_M5IL);
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
function do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid,emld_config,response_path,feature_path,struct_M5IL_config)

% fprintf('coding split:%d class:%d vid:%d feature:%s\n',cur_splits,cur_class,cur_vid,cur_feature);

load_name = fullfile(feature_path,sprintf('c%03d_v%03d_info.mat',cur_class,cur_vid));
load(load_name,'vid_info');

switch cur_feature(end)
    
    case '1'
        load_name = fullfile(response_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature));
        load(load_name,'responds_data','responds_location');
        
       
        
        switch emld_config.params.pooling_range_type
            
            case 'video'
                pooling_range = [vid_info(1),vid_info(1),vid_info(2),vid_info(2),vid_info(3),vid_info(3)];
                pooling_cells = emld_config.params.pooling_cells.*...
                    (repmat(pooling_range,size(emld_config.params.pooling_cells,1),1));
                
            case 'bbox'
                min_loc = min(responds_location(:,1:3),[],1);
                delta_loc = max(responds_location(:,1:3),[],1) - min_loc;
                
                low_range = [min_loc(2),min_loc(2),min_loc(3),min_loc(3),min_loc(1),min_loc(1)];
                delta_range = [delta_loc(2),delta_loc(2),delta_loc(3),delta_loc(3),delta_loc(1),delta_loc(1)];
                
                pooling_cells = repmat(low_range,size(emld_config.params.pooling_cells,1),1) + ...
                    emld_config.params.pooling_cells.* repmat(delta_range,size(emld_config.params.pooling_cells,1),1);
                
            otherwise
                fprintf('WRONG!!! the type of pooling range %s not exist!\n ',emld_config.params.pooling_range);
                
        end
        
        pooling_cells_num = size(pooling_cells,1);
        
        
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
   
    
    otherwise
        
        load_name = fullfile(response_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature));
        load(load_name,'phi_nodes_all','phi_edges_all','parts_location');
        
        responds_location = parts_location;
        
        switch emld_config.params.pooling_range_type
            
            case 'video'
                pooling_range = [vid_info(1),vid_info(1),vid_info(2),vid_info(2),vid_info(3),vid_info(3)];
                pooling_cells = emld_config.params.pooling_cells.*...
                    (repmat(pooling_range,size(emld_config.params.pooling_cells,1),1));
                
            case 'bbox'
                min_loc = min(responds_location(:,1:3),[],1);
                delta_loc = max(responds_location(:,1:3),[],1) - min_loc;
                
                low_range = [min_loc(2),min_loc(2),min_loc(3),min_loc(3),min_loc(1),min_loc(1)];
                delta_range = [delta_loc(2),delta_loc(2),delta_loc(3),delta_loc(3),delta_loc(1),delta_loc(1)];
                
                pooling_cells = repmat(low_range,size(emld_config.params.pooling_cells,1),1) + ...
                    emld_config.params.pooling_cells.* repmat(delta_range,size(emld_config.params.pooling_cells,1),1);
                
            otherwise
                fprintf('WRONG!!! the type of pooling range %s not exist!\n ',emld_config.params.pooling_range);
                
        end
        
        pooling_cells_num = size(pooling_cells,1);
        channel_num = size(phi_nodes_all,1);
        W_num = size(phi_nodes_all,2);
        parts_num = size(phi_nodes_all{1,1},1);
        
        
        % pooling --------------------
        ml_descriptor = [];
        for i_cell = 1:pooling_cells_num
            
            select_id = (responds_location(:,2) >= pooling_cells(i_cell,1))&(responds_location(:,2) <= pooling_cells(i_cell,2))&...
                (responds_location(:,3) >= pooling_cells(i_cell,3))&(responds_location(:,3) <= pooling_cells(i_cell,4))&...
                (responds_location(:,1) >= pooling_cells(i_cell,5))&(responds_location(:,1) <= pooling_cells(i_cell,6));
            
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
            
            
            responds_data = zeros(W_num,channel_num);
            
            if(~isempty(select_nodes_id))
              
                for i_W = 1:size(phi_nodes_all,2)
                    
                    phi_nodes_in_cell = cell(channel_num,1);
                    phi_edges_in_cell = cell(channel_num,1);
                    for i_channel = 1:channel_num
                        
                        phi_nodes_in_cell{i_channel} = phi_nodes_all{i_channel,i_W}(select_nodes_id,:);
                        phi_edges_in_cell{i_channel} = phi_edges_all{i_channel,i_W}(select_edges_id,:);
                        
                    end
                    
                    responds_data(i_W,:) = infer_bag_score(phi_nodes_in_cell,phi_edges_in_cell,struct_M5IL_config);
                end
              
            end
            % infer in the cell
           
            
            responds_data = reshape(responds_data',1,[]);
            ml_descriptor = [ml_descriptor,responds_data];
            
        end
        
        if sum(sum(isnan(ml_descriptor)))
            printf('WRONG!! NaN occur!\n');
        end
        
       
        save_name = fullfile(emld_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
        save(save_name,'ml_descriptor');
        
   
end


end



%%
% output:
% scores [1,channel_num]
% psi_max_score, the related psi of the max score channel
function [scores,node_ass_idx] = infer_bag_score(phi_nodes_all,phi_edges_all,config)

switch config.method
    case 'TRW-S'
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i_channel = 1:config.channel_num
            phi_nodes_all{i_channel} = -phi_nodes_all{i_channel};
            phi_edges_all{i_channel} = -phi_edges_all{i_channel};
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