% extract part data from trajectory features, grouping method are: VOI or
% cluster
% output: c%03d_v%03d_[feature_type].mat
% c%03d_v%03d_location.mat
%

% v1: pooling on the addaptive VOI
% v2: add parts_pair_feature,but all zeros
% v3: real parts_pair_features
% v4: suppot not syn pair
function return_state = m_extract_part_features_v3_vc(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

extract_part_features_config = global_config.extract_part_features;
splits = extract_part_features_config.splits;
feature_types = extract_part_features_config.feature_types;

do_sets = global_config.extract_part_features.class_idx;

do_imgs = cell(length(splits),class_num);

switch class(do_sets)
    case 'cell'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_imgs{cur_splits,do_sets{i}(1)} = [do_imgs{cur_splits,do_sets{i}(1)};do_sets{i}(2)];
            end
            
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'double' %splits = 1, do_sets = 1:15; test_train_set_idx load from data
        
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_imgs{cur_splits,do_sets(i)} = [train_set_idx{cur_splits}{do_sets(i)} test_set_idx{cur_splits}{do_sets(i)}]';
            end
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
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
    
    for i_class = 1:class_num
        
        cur_class = i_class;
        if isempty(do_imgs{cur_splits,i_class})
            continue;
        end
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};
            
            fprintf('extract part feature split:%2d class:%3d  feature:%s\n',cur_splits,cur_class,cur_feature);
            % all data
            cur_img_idx = do_imgs{cur_splits,i_class};
            
            switch extract_part_features_config.params.method
                case 'cluster'
                    
                    if global_config.num_core > 1
                        parfor i_img = 1:length(cur_img_idx)
                            %do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_img_idx(i_img),class_names, img_names,extract_part_features_config,...
                            %    global_config.coding.path,global_config.extract_features.path,global_config.clustering.path);
                            do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_img_idx(i_img),class_names, img_names,extract_part_features_config,...
                                global_config.coding.path,global_config.vc_path,global_config.clustering.path);
                        end
                    else
                        for i_img = 1:length(cur_img_idx)
                            do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_img_idx(i_img),class_names, img_names,extract_part_features_config,...
                                global_config.coding.path,global_config.vc_path,global_config.clustering.path);
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
function do_pooling_cluster_max(cur_splits, cur_feature, cur_class, cur_img, class_names, img_names, extract_part_features_config, codes_path,feature_path,clusters_path)

% fprintf('extract part feature split:%2d class:%3d vid:%3d feature:%s\n',cur_splits,cur_class,cur_vid,cur_feature);

load_name = fullfile(clusters_path,sprintf('clusters_c%03d_v%03d.mat',cur_class,cur_img));
load(load_name,'cluster_id');

switch class(cluster_id)
    case 'double'
        cluster_num = max(cluster_id);
    case 'cell'
        cluster_num = length(cluster_id);
    otherwise
        fprintf('WRONG!!! type of cluster_id wrong\n');
end

if strcmp(cur_feature,'pair')
    
% load vc feature //Because all zero so take it out
%load_name = fullfile(feature_path, class_names{cur_class}, [img_names{cur_class}{cur_img}(1:end-4)]);
%load(load_name);
%load_name = fullfile(feature_path,sprintf('c%03d_v%03d_trajectory.mat',cur_class,cur_img));
%load(load_name,'trajectory');

% compute pair feature ---------------------------------------
edge_dim = 100;

cluster_total = 0;
for i_cluster = 1:cluster_num
     cluster_total = cluster_total + max(cluster_id{i_cluster});   
end
cluster_num = cluster_total;
parts_pair_features = zeros(cluster_num*(cluster_num+1)/2,edge_dim);

ind = 0;
for i = 1:cluster_num
    for j = i:cluster_num
        % compute cluster pair feature
        ind = ind +1;
        parts_pair_features(ind,:) = zeros(1,edge_dim);        
    end
end
save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_pair.mat',cur_splits,cur_class,cur_img) );
save(save_name,'parts_pair_features');

% -------------------------------------------------------------------------

else
    %%% MEANS ONLY TO EXTRACT LOCATION, REMOVE THEM ALL
    % load vc feature
load_name = fullfile(feature_path, class_names{cur_class}, [img_names{cur_class}{cur_img}(1:end-4)]);
load(load_name);

cluster_total = 0;
for i_cluster = 1:cluster_num
     cluster_total = cluster_total + max(cluster_id{i_cluster});   
end

parts_features = zeros(cluster_total,size(response,1));
parts_location = zeros(cluster_total,2);

cluster_count = 0;
for i_cluster = 1:cluster_num
   for c_cluster = 1:max(cluster_id{i_cluster})
  
    cluster_count = cluster_count + 1;
    switch class(cluster_id)
        case 'double'
            select_id = find(cluster_id == i_cluster);
        case 'cell'
            select_id = (cluster_id{i_cluster} == c_cluster);
        otherwise
            fprintf('WRONG!!! type of cluster_id wrong\n');
    end
    
    linearid = find(select_id==1);
    [I,J] = ind2sub(feasizes,find(select_id==1));
    cur_location = [I;J]';
    %cur_location = location(select_id,:);
    
    x_low = min(cur_location(:,1));
    x_high = max(cur_location(:,1));
    
    y_low = min(cur_location(:,2));
    y_high = max(cur_location(:,2));
    %
    %t_low = min(cur_location(:,1));
    %t_high = max(cur_location(:,1));
    %
    %VOI_select_id = (location(:,2) >= x_low)&(location(:,2) <= x_high)&...
    %    (location(:,3) >= y_low)&(location(:,3) <= y_high )&...
    %    (location(:,1) >= t_low)&(location(:,1) <= t_high);
    cur_codes = response(:,select_id);   
    %cur_codes = codes(VOI_select_id,:);
    
    
    %max_loc = max(cur_location(:,1:),[],1);
    %min_loc = min(cur_location(:,1:3),[],1);
    
   % parts_location(i_cluster,:) = mean(cur_location(:,1:3),1);
   
   parts_location(cluster_count,:) = [(x_low+x_high)/2,(y_low+y_high)/2];
    
% % %     switch extract_part_features_config.params.pooling
% % %         case 'max'
% % %             parts_features(cluster_count,:) = max(cur_codes,[],2);
% % %             
% % %         otherwise
% % %              fprintf('wrong type of extract_part_features_config.params.pooling %s!\n',extract_part_features_config.params.pooling);
% % %              returnl
% % %                 
% % %     end
   end
    
end


% normalization using l2 ----------------------
%%%EPS = 1e-6;
%%%norm_data = sqrt( sum(parts_features.^2,2 ) );
%%%parts_features(norm_data >EPS,:) = parts_features(norm_data >EPS,:) ./ repmat(norm_data(norm_data>EPS),1,size(parts_features,2));

%%%save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_img,cur_feature) );
%%%save(save_name,'parts_features');

save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_location.mat',cur_splits,cur_class,cur_img) );
save(save_name,'parts_location');

end


end