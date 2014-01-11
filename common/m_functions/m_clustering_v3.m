
% v2: add hirarchi cluster
% v3: add soft cluster
function return_state = m_clustering_v3(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

clustering_config = global_config.clustering;

do_class = global_config.clustering.class_idx;
do_vids = cell(class_num,1);
switch class(do_class)
    case 'cell'
        for i = 1:length(do_class)
            do_vids{do_class{i}(1)} = [do_vids{do_class{i}(1)};do_class{i}(2)];
        end
    case 'double'
        for i = 1:length(do_class)
            do_vids{do_class(i)} = 1:vid_nums_in_class(do_class(i));
        end
        
    case 'char'
        for i = 1:class_num
            do_vids{i} = 1:vid_nums_in_class(i);
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end

for i_class = 1:class_num
    cur_class = i_class;
   
    if length(do_vids{i_class}) == 0
        continue;
    end 
    tic;
    cur_class_vids = do_vids{i_class};
    if global_config.num_core > 1
        parfor i_vid = 1: length(cur_class_vids)
            do_clustering_v3(cur_class,cur_class_vids(i_vid),clustering_config,global_config.extract_features.path)
        end
    else
        for i_vid = 1: length(cur_class_vids)
           do_clustering_v3(cur_class,cur_class_vids(i_vid),clustering_config,global_config.extract_features.path)
        end
    end
    toc;
end

end


