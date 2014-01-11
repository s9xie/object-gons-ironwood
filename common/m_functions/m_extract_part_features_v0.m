% extract part data from trajectory features, grouping method are: VOI or
% cluster
% output: c%03d_v%03d_[feature_type].mat 
% c%03d_v%03d_location.mat
% 


function return_state = m_extract_part_features_v0(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

extract_part_features_config = global_config.extract_part_features;
splits = extract_part_features_config.splits;
feature_types = extract_part_features_config.feature_types;

do_sets = global_config.extract_part_features.class_idx;

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
        if length(do_vids{cur_splits,i_class}) == 0
            continue;
        end
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};

            fprintf('extract part feature split:%2d class:%3d  feature:%s\n',cur_splits,cur_class,cur_feature);
            % all data
            cur_vid_idx = do_vids{cur_splits,i_class};
            
            switch extract_part_features_config.params.method
                case 'cluster'
                  
                     if global_config.num_core > 1
                        parfor i_vid = 1:length(cur_vid_idx)
                            do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),extract_part_features_config,...
                                global_config.coding.path,global_config.extract_features.path,global_config.clustering.path);
                        end
                    else
                        for i_vid = 1:length(cur_vid_idx)
                            do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),extract_part_features_config,...
                                global_config.coding.path,global_config.extract_features.path,global_config.clustering.path);
                        end
                    end
                                       
                otherwise
                    fprintf('wrong type of extract_part_features_config.params.method %s!\n',extract_part_features_config.params.method);
                    return_state = 0;
                    return;
            end
            
        end  % ifeature_type
    end % iclass ends
    
end % i_split

end
%%

function do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_vid,extract_part_features_config, codes_path,feature_path,clusters_path)

% fprintf('extract part feature split:%2d class:%3d vid:%3d feature:%s\n',cur_splits,cur_class,cur_vid,cur_feature);

load_name = fullfile(codes_path,sprintf('codes_s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature));
load(load_name,'codes');

load_name = fullfile(feature_path,sprintf('c%03d_v%03d_info.mat',cur_class,cur_vid));
load(load_name,'vid_info');

load_name = fullfile(feature_path,sprintf('c%03d_v%03d_location.mat',cur_class,cur_vid));
load(load_name,'location');

load_name = fullfile(clusters_path,sprintf('clusters_c%03d_v%03d.mat',cur_class,cur_vid));
load(load_name,'cluster_id');

cluster_num = max(cluster_id);

parts_features = zeros(cluster_num,size(codes,2));
parts_location = zeros(cluster_num,3);
for i_cluster = 1:cluster_num
   
    
    select_id = find(cluster_id == i_cluster);
    cur_codes = codes(select_id,:);
    cur_location = location(select_id,:);
    
%     max_loc = max(cur_location(:,1:3),[],1);
%     min_loc = min(cur_location(:,1:3),[],1);
    
    parts_location(i_cluster,:) = mean(cur_location(:,1:3),1);
    
    switch extract_part_features_config.params.pooling
        case 'max'
            parts_features(i_cluster,:) = max(cur_codes,[],1);
            
        otherwise
             fprintf('wrong type of extract_part_features_config.params.pooling %s!\n',extract_part_features_config.params.pooling);
             returnl
                
    end
    
end

save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
save(save_name,'parts_features');

save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_location.mat',cur_splits,cur_class,cur_vid) );
save(save_name,'parts_location');


end