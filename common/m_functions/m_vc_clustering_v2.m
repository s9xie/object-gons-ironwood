% wyw 9/6/2013 @MSRA
% input: trajectories (the location of each trajectory)
% output: clusters_c%03d_v%03d.mat, cluster_id of the trajectories in each clusters


function return_state = m_vc_clustering_v2(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);
vc_path = global_config.vc_path;

clustering_config = global_config.vc_clustering;

do_class = global_config.vc_clustering.class_idx;
do_imgs = cell(class_num,1);
switch class(do_class)
    case 'cell'
        for i = 1:length(do_class)
            do_imgs{do_class{i}(1)} = [do_imgs{do_class{i}(1)};do_class{i}(2)];
        end
    case 'double'
        for i = 1:length(do_class)
            do_imgs{do_class(i)} = 1:img_nums_in_class(do_class(i));
        end
        
    case 'char'
        for i = 1:class_num
            do_imgs{i} = 1:vid_nums_in_class(i);
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end

for i_class = 1:class_num
    cur_class = i_class;
   
    if isempty(do_imgs{i_class})
        continue;
    end 
    tic;
    cur_class_imgs = do_imgs{i_class};
    if global_config.num_core > 1
        parfor i_img = 1: length(cur_class_imgs)
            %do_vc_clustering_v2(cur_class,cur_class_imgs(i_img),clustering_config,global_config.extract_features.path)
            do_vc_clustering_v2(cur_class,cur_class_imgs(i_img), class_names, img_names, clustering_config, vc_path);
            
        end
    else
        for i_img = 1: length(cur_class_imgs)
           %do_vc_clustering_v2(cur_class,cur_class_imgs(i_img),clustering_config,global_config.extract_features.path)
           do_vc_clustering_v2(cur_class,cur_class_imgs(i_img), class_names, img_names, clustering_config, vc_path);
        end
    end
    toc;
end

end


