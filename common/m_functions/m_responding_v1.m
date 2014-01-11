%
% add response analyse
%

function return_state = m_responding_v1(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);
responding_config = global_config.responding;

% generate the splits and feature types according to global_config
splits = responding_config.splits;
feature_types = responding_config.feature_types;


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
        W = [];
        for i_class = 1:class_num
            
            cur_class = i_class;
            if length(do_vids{cur_splits,cur_class}) == 0
                continue;
            end
            
            load_name = fullfile(global_config.learn_action_parts.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,cur_class,cur_feature));
            load( load_name, 'action_parts_model');
            
            W = [W,action_parts_model];
        end
        
        if sum(sum(isnan(W)))
           
           fprintf('WRONG!!! NaN occur! splits:%d feature:%s\n ',cur_splits,cur_feature);
           return_state = 0;
           return;
           
        end
        
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
                    do_responding(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),W,global_config.extract_part_features.path,responding_config.path);
                end
            else
                for i_vid = 1:length(cur_vid_idx)
                    do_responding(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),W,global_config.extract_part_features.path,responding_config.path);
                end
            end
            
            
            
        end  % iclass ends
    end  % ifeature_type
    
end % i_split

end

%%
function do_responding(cur_splits,cur_feature,cur_class,cur_vid,W,part_features_path,responding_path)

EPS = 1e-6;
% load instance feature data
load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_location.mat',cur_splits,cur_class,cur_vid) );
load(load_name,'parts_location');

load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
load(load_name,'parts_features');

responds_data = [parts_features,ones( size(parts_features,1) ,1)] * W;
if sum(sum( isnan(responds_data) ))
    fprintf('WRONG!!! NaN occur in responsing! %s\n',load_name);
end

cur_class_instaces{i_vid,i_feature} = (parts_features);

% [response_sort,response_idx] = sort(responds_data,2,'descend');

responds_location  = parts_location;
save_name = fullfile(responding_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
save(save_name,'responds_data','responds_location');


end
