function return_state = m_extract_mid_level_descriptors_v0(global_config)
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
                            do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),emld_config,global_config.responding.path,global_config.extract_features.path);
                        end
                    else
                        for i_vid = 1:length(cur_vid_idx)
                            do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),emld_config,global_config.responding.path,global_config.extract_features.path);
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
function do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid,emld_config,response_path,feature_path)

% fprintf('coding split:%d class:%d vid:%d feature:%s\n',cur_splits,cur_class,cur_vid,cur_feature);

load_name = fullfile(response_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature));
load(load_name,'responds_data','responds_location');

load_name = fullfile(feature_path,sprintf('c%03d_v%03d_info.mat',cur_class,cur_vid));
load(load_name,'vid_info');

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

pooling_cells_num = length(pooling_cells);


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

ml_descriptor = 1./(1+exp(-emld_config.params.rho*ml_descriptor));

save_name = fullfile(emld_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
save(save_name,'ml_descriptor');

end